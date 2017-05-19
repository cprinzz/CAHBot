var natural = require('natural')
var jd = require('jsdataframe')
var fs = require('fs')
var path = require('path')
var _ = require('underscore')

//TODO:
// [] Get sort to work
// [] Check against py script
var CAHBot = function ()
{
    this.GloveVectors = {};
    this.GloveVecFile = 'glove.6B.50d.txt';
    this.CachedVectors = {}
    this.CachedVecFile = 'vectors.json';
    this.Cards = {};

    this.readCards = function(){
        // {blackCards:[{text:str, pick:int}],
        //  whiteCards:[{text:str, keepCap: bool, tags:[str]}]}
        var allText = fs.readFileSync('cards.json', 'utf8')
        this.Cards = JSON.parse(allText)
    };

    this.readCachedVectors = function(){
        // Reads CachedVecFile and stores vectors in CachedVectors
        allText = fs.readFileSync(this.CachedVecFile, 'utf8')
        var word_vectors = JSON.parse(allText)
        this.CachedVectors = word_vectors
    };

    this.readGloveVectors = function(){
        // Reads GloveVecFile and stores vectors in GloveVectors
        allText = fs.readFileSync(this.GloveVecFile, 'utf8')
        allText = allText.split('\n')
        for (var line = 0; line < allText.length; line++){
          splitLine = allText[line].split(' ')
          word = splitLine[0]
          vector = []
          for (var num = 1; num < splitLine.length; num++){
            vector.push(parseFloat(splitLine[num]))
          }
          this.GloveVectors[word] = vector
        }
        console.log("Done." + Object.keys(this.GloveVectors).length + " words loaded!")
    };

    this.addWordsToCache = function(word, vector){
        this.CachedVectors[word] = vector
        fs.readFile(this.CachedVecFile,function(err,content){
            if (err) {
                throw err
            }
            var parseJson = JSON.parse(content)
            parseJson[word] = vector
            fs.writeFile('data.json',JSON.stringify(parseJson),function(err){
                if (err) {
                    throw err
                }
            })
        })
    };

    this.createSentenceVector = function(text){
        // Params: sentence string
        // Returns: matrix of word vectors
        var tokenizer = new natural.WordTokenizer()
        var stopwords = natural.stopwords
        var base_folder = path.join(path.dirname(require.resolve("natural")), "brill_pos_tagger");
        var rulesFilename = base_folder + "/data/English/tr_from_posjs.txt";
        var lexiconFilename = base_folder + "/data/English/lexicon_from_posjs.json";
        var defaultCategory = 'N';

        var lexicon = new natural.Lexicon(lexiconFilename, defaultCategory);
        var rules = new natural.RuleSet(rulesFilename);
        var tagger = new natural.BrillPOSTagger(lexicon, rules);

        var tokens = tokenizer.tokenize(text)
        var vectors = []

        for (var i = 0; i < Object.keys(tokens).length; i++ ){
            token = tokens[i].toLowerCase()
            if (stopwords.indexOf(token) == -1){
                if (Object.keys(this.CachedVectors).indexOf(token) != -1){
                    if (tagger.tag([token])[1] == 'N' || tagger.tag([token]) == 'VB'){
                        var vector = this.CachedVectors[token].map(function(x) {x * 1.5})
                        vectors.push(vector)
                    }
                    else{
                        vectors.push(this.CachedVectors[token])
                    }
                }
                else {
                    if (this.GloveVectors.length == 0){
                        this.readGloveVectors()
                    }
                    if (token in this.GloveVectors){
                        vector = this.GloveVectors[token]
                        vectors.push(vector)
                        this.addWordsToCache(token, vector)
                    }
                }
            }
        }
        //If vectors is empty (no words found), return vector of zero vec dim (50,0)
        if (vectors.length == 0){
            return jd.rep(0,50)
        }
        return vectors
    };

    this.createAverageVector = function(sentence_mat){
        // Params: matrix of word vectors
        // Returns: average vector of sentence matrix
        sum_vec = jd.rep(0,50)
        for (var i = 0; i < sentence_mat.length; i++ ){
            var vec = sentence_mat[i]
            sum_vec = sum_vec.add(vec)
        }

        avg_vec = sum_vec.div(sentence_mat.length)
        return avg_vec
    };

    this.cosineSimilarity = function(v1, v2){
        // compute cosine similarity of v1 to v2: (v1 dot v2)/{||v1||*||v2||)
        var sumxx = 0
        var sumxy = 0
        var sumyy = 0
        for ( var i = 0; i < 50; i++ ){
            var x = v1[i]
            var y = v2[i]
            sumxx += x*x
            sumyy += y*y
            sumxy += x*y
        }
        return sumxy/Math.sqrt(sumxx*sumyy)
    };

    //Pick Best card
    this.pickBestCard = function(wc_array, black_card){
        // """
        // Params: Array of white card dicts, black card text
        // in the format {text: string, pick: int}
        // Returns: dict of best card texts, average vectors, and confidence scores
        //
        // If wc_array = None, pick best white card from entire set
        // """
        var wc_vectors = {}
        // Creates an array of word vectors for each wc text
        for (var i = 0; i < wc_array.length; i++ ){
            var wc_dict = wc_array[i]
            wc_vectors[wc_dict.text] = this.createSentenceVector(wc_dict.text)
        }
        // Creates list of vectors for black card text
        var bc_vector = this.createSentenceVector(black_card)

        // Iterates through each cards vectors. Adds each array elementwise and
        // and takes the average.
        var avg_dict = {}

        for (var i = 0; i < wc_array.length; i++ ){
            var wc_dict = wc_array[i]
            avg_dict[wc_dict.text] = this.createAverageVector(wc_vectors[wc_dict.text])
        }
        // Create black card avg vector
        var bc_avg_vector = this.createAverageVector(bc_vector)
        // Create cosine similarity scores for each prompt/response combination
        var prompt = black_card
        var prompt_vec = bc_avg_vector.values
        var scores = [] //{text:score}
        for (var i = 0; i < Object.keys(avg_dict).length; i++ ){
            var key = Object.keys(avg_dict)[i]
            var response_vec = avg_dict[key].values
            var similarity = this.cosineSimilarity(prompt_vec, response_vec)
            var abs_distance_from_zero = parseFloat(Math.abs(similarity))
            console.log(key)
            var score = {}
            score[key] = abs_distance_from_zero
            scores.push(score)
        }

        //Sort array of dicts by values ascending
        //TODO: get sort to work
        scores.sort(function(a,b){
            return a.value - b.value
        })
        console.log(scores)
        return Object.keys(scores[0])
    };

    this.formResponse = function(black_card, wc_array){
        // """
        // Params: array of white card dicts, black_card dict, num picks
        // Returns: Complete sentence replacing '_' in black card text with
        //          responses
        // """

        if (wc_array == undefined){
            this.readCards()
            wc_array = this.Cards.whiteCards
        }

        while (black_card.indexOf('_') != -1){
            var wc_text = this.pickBestCard(wc_array, black_card)
            console.log(wc_text)
            // Get dict to pop later
            var wc_dict = {}
            wc_array.forEach(function(item){
                if (item.text == wc_text){
                    wc_dict = item
                }
            })
            // Clean text
            wc_dict.text = wc_dict['text'].replace(/[^\w\s]|_/g, "")
            if(wc_dict.keepCap == false){
                wc_dict.text = wc_dict['text'].toLowerCase()
            }
            black_card = black_card.replace('_', wc_dict.text)
            wc_array.pop(wc_array.indexOf(wc_dict))
        }
        return black_card
    };

    this.simulateRound = function(pick){
        this.readCards()
        var cards = this.Cards
        // Get random 7 response cards
        var wc_array = _.sample(this.Cards.whiteCards, 7)
        // Get 1 random black card with pick = x
        var pick_x = []
        this.Cards.blackCards.forEach(function(card){
            if (card.pick == pick){
                pick_x.push(card)
            }
        })
        var black_card = _.sample(pick_x, 1)[0]['text']

        if (black_card.indexOf('_') == -1){
            black_card += ' _'
        }

        var response = this.formResponse(black_card, wc_array)
        return response
    };

}

var test = new CAHBot()
test.readCachedVectors()
console.log(test.simulateRound(1))
