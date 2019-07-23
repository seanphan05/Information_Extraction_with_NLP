###############################################################################

import PyPDF2, pprint, docx, nltk
import pandas as pd
import numpy as np
import random
import csv, sqlite3
from nltk import word_tokenize, pos_tag
from nltk import conlltags2tree, tree2conlltags
from nltk import ChunkParserI, TaggerI, NaiveBayesClassifier, RegexpParser
from nltk import grammar, parse
from nltk.stem import WordNetLemmatizer
from nltk.tree import Tree
from segment_sents import segment_sents

###############################################################################
###############################################################################

# function to extract text from pdf
def pdf_extract(pdf_file, first_page, last_page):
    pdf_reader = PyPDF2.PdfFileReader(pdf_file) 
    pdf_text = []
    for pageNum in range(first_page, last_page):
        pdf_obj = pdf_reader.getPage(pageNum)
        pdf_text.extend([pdf_obj.extractText()])
    return (pdf_text)

# function to process raw text extracted from MOU file
def process_text(text):
    token_text = []
    clean_text = []
    lemmatizer = WordNetLemmatizer()
    num_syl = '''%$'''
    token_text.extend(word_tokenize(i) for i in text)
    tokenized_text = sum(token_text, [])
    lem_text = [lemmatizer.lemmatize(word) for word in tokenized_text]
    seg_sents = segment_sents(lem_text)
    for sent in seg_sents:
        clean_sent = [clean.lower() for clean in sent if len(clean)>1 or clean in num_syl]
        clean_text.append(clean_sent)
    return (clean_text)

# function to create dataframe for POS tagged sentences
def df_pos(tree):
    pos_df = pd.DataFrame(columns = ['Word', 'POS'])
    for i in range(0,len(tree)):
        df = pd.DataFrame(tree[i], columns = ['Word', 'POS'])
        num_sent = pd.DataFrame(data=i+1, index=np.arange(len(tree[i])), columns = ['Sentence #'])
        data_sent = pd.concat([num_sent, df], axis=1, sort=False)
        pos_df = pos_df.append(data_sent, ignore_index=True)
    pos_df = pos_df.reindex(['Sentence #','Word', 'POS'], axis='columns')
    return (pos_df)

# function to create dataframe for IOB tagged sents
def df_iob(iob_tags):
    data = pd.DataFrame(columns = ['Word', 'POS', 'IOB Tag'])
    for i in range(0, len(iob_tags)):
        df = pd.DataFrame(iob_tags[i], columns = ['Word', 'POS', 'IOB Tag'])
        num_sent = pd.DataFrame(data='sentence %s' % (i+1), index=np.arange(len(iob_tags[i])), 
                            columns = ['Sentence'])
        data_sent = pd.concat([num_sent, df], axis=1, sort=False)
        data = data.append(data_sent, ignore_index=True)
    data = data.reindex(['Sentence','Word', 'POS', 'IOB Tag'], axis='columns')
    return (data)

# functions to flatten deep-trees
def flatten_childtrees(trees): 
    children = []
    for t in trees:
        if isinstance(t, Tree):
            if t.height() <= 3:
                children.append(Tree(t.label(), [i[0] for i in t.pos()]))
            else:
                children.extend(flatten_childtrees(t))
        else:
            children.append(t)
    return children
def flatten_deeptree(tree): 
    return Tree(tree.label(), flatten_childtrees(tree))


# class for classifier-based chunk tagger
class ChunkTagger(TaggerI):
    def __init__(self, train_sents):
        train_set = []
        for tagged_sent in train_sents:
            untagged_sent = nltk.tag.untag(tagged_sent)
            history =[]
            for i, (word, tag) in enumerate(tagged_sent):
                featureset = chunk_features(untagged_sent, i, history)
                train_set.append((featureset, tag))
                history.append(tag)
        self.classifier = NaiveBayesClassifier.train(train_set)
    
    def tag(self, sentence):
        history = []
        for i, word in enumerate(sentence):
            featureset = chunk_features(sentence, i, history)
            tag = self.classifier.classify(featureset)
            history.append(tag)
        return (zip(sentence, history))
    

# class for chunk classifier
class ChunkClassifier(ChunkParserI):
    def __init__(self, train_sents):
        # convert the triplets to pairs to comply with tagger format
        tagged_sents = [[((w, t), c) for (w, t, c) in tree2conlltags(sent)] for sent in train_sents]
        self.feature_detector = chunk_features
        self.tagger = ChunkTagger(tagged_sents)
    
    def parse(self, sent):
        tagged_sents = self.tagger.tag(sent)
        iob_sents = [(w, t, c) for ((w, t), c) in tagged_sents]
        return (conlltags2tree(iob_sents))


# function for feature extraction
def chunk_features(tokens, index, history):
    global preword, prepos, nextword, nextpos
    word, pos = tokens[index]
    
    if index == 0:
        preword, prepos = "<START>", "<START>"
        prex2word, prex2pos = "<START-1>", "<START-1>"
    elif index == 1:
        prex2word, prex2pos = "<START>", "<START>"
    else:
        preword, prepos = tokens[index-1]
        prex2word, prex2pos = tokens[index-2]

    if index == len(tokens)-1:
        nextword, nextpos = "<END>", "<END>"
        nextx2word, nextx2pos = "<END+1>", "<END+1>"
    elif index == len(tokens)-2:
        nextx2word, nextx2pos = "<END>", "<END>"
    else:
        nextword, nextpos = tokens[index+1]
        nextx2word, nextx2pos = tokens[index+2]
        
    return {"word": word,
            "pos": pos,
            
            "preword": preword,
            "prepos": prepos,
            "2nd-preword": prex2word,
            "2nd-prepos": prex2pos,
            
            "nextword": nextword,
            "nextpos": nextpos,
            "2nd-nextword": nextx2word,
            "2nd-nextpos": nextx2pos,
            
            "prepos/pos": "%s+%s" % (prepos, pos),
            "pos/nextpos": "%s+%s" % (pos, nextpos)}

###############################################################################
###############################################################################

# load and open ALL_MOU.pdf file
MOU_file = open('All_MOUs.pdf', 'rb')
# extract compensation sections text of 43 MOUs (except MOU1) from ALL_MOU.pdf
# pages list contains compensation information
comp_pages = [(175,187),(271,280),(347,359),(567,578),(670,682),(787,801),
              (842,853),(895,909),(1067,1078),(1287,1290),(1422,1427),(1620,1633),
              (1808,1818),(2067,2070),(2200,2202),(2405,2418),(2661,2668),(2842,2853),
              (3089,3100),(3288,3296),(3501,3514),(3908,3921),(4111,4124),(4351,4365),
              (4598,4609),(4725,4746),(4896,4921),(5218,5221),(5336,5338),(5457,5463),
              (5578,5584),(5832,5837),(6161,6164),(6301,6307),(6403,6406),(6618,6620),
              (6826,6832),(6975,6978),(7075,7077),(7144,7152),(7235,7238)]
# extract compensation information of 43 MOUs
MOU_comp = []
for i in comp_pages:
    MOU_text = pdf_extract(MOU_file, i[0], i[1])
    MOU_text = list(map(lambda x: x.replace('\n',''),MOU_text))
    MOU_comp.extend(MOU_text)

# process raw data
MOU_sents = process_text(MOU_comp)
# tag sentences with pos_tag
tree = [pos_tag(i) for i in MOU_sents]

# export to csv file for manual POS tags correction
pos_df = df_pos(tree)
pos_df.to_csv('pos_data.csv')

# import pos_df with correct pos tags to csv file
new_pos_df = pd.read_csv('cor_pos_data.csv')
num_of_sent = len(tree)
new_tree = []
for i in range(1, (num_of_sent + 1)):
    sent_data = new_pos_df[new_pos_df['Sentence #'] == i]
    sent_tuple = [tuple(word) for word in sent_data[['Word','POS']].values]
    new_tree.append(sent_tuple) 

# chunk sentences with regular expression-based approach
rule = r"""
  NP:     {<DT|RB|PRP\$|JJ|NN><JJ>*<NN>+}           # Chunk noun phrases
  VP:     {<MD|VB><VB>*<JJ>*<IN|TO>?<VB>?}          # Chunk verb phrases
  NUM:    {<\$><CD><IN>*}                           # Chunk numbers
          {<CD><\%><IN>*}
"""
chunkParser = RegexpParser(rule)
tree_tags = [chunkParser.parse(i) for i in new_tree]
# flatten deep trees for iob tagging
flat_tree = [flatten_deeptree(i) for i in tree_tags]

# training the chunker
# prepare training and testing data
random_sample = random.sample(flat_tree, len(flat_tree))
train_data = random_sample[:int(len(random_sample)*0.8)]
test_data = random_sample[int(len(random_sample)*0.8 + 1):]
# evaluate classifier chunker
classifier_chunker = ChunkClassifier(train_data)
print(classifier_chunker.evaluate(test_data))

# add iob tags
iob_tags = [tree2conlltags(i) for i in flat_tree]
# create annotated database and export to csv file
iob_df = df_iob(iob_tags)
iob_df.to_csv('model-building_dataset.csv')

###############################################################################
###############################################################################

# apply chunk classifier on MOU1 compensation corpus
# read MOU1 compensation corpus 
def getText(filename):
    doc = docx.Document(filename)
    fullText = []
    for para in doc.paragraphs:
        fullText.append(para.text)
    return '\n'.join(fullText)
MOU1_text = getText('MOU1_Compensation.docx')
MOU1_line = MOU1_text.split('\n')
# split text by paragraphs
MOU1_para = []
lines = []
for line in MOU1_line:
    if len(line.strip()) != 0:
        lines.append(line)
    else:
        MOU1_para.append(lines)
        lines = []

# extract compensation categories info
MOU1_article = []
for i in MOU1_para:
    article_name = ' '.join(i.pop(0).split()[2:]) 
    MOU1_article.append(article_name)
print(MOU1_article)

# process MOU1 text raw data
MOU1_doc = [process_text(sent) for sent in MOU1_para]

# tag sentences with pos_tag
tree_MOU1 = []
for doc in MOU1_doc:
    tree_doc = [pos_tag(i) for i in doc]
    tree_MOU1.append(tree_doc)

# export to csv file for manual POS tag correction
MOU1_pos_df = pd.DataFrame(columns = ['Sentence #', 'Word', 'POS'])
for i in range(0, len(tree_MOU1)):
    df = df_pos(tree_MOU1[i])
    num_arti = pd.DataFrame(data=i+1, index=np.arange(len(df)), columns = ['Article #'])
    MOU1_pos = pd.concat([num_arti, df], axis=1, sort=False)
    MOU1_pos_df = MOU1_pos_df.append(MOU1_pos, ignore_index=True)
    MOU1_pos_df = MOU1_pos_df.reindex(['Article #','Sentence #','Word', 'POS'], axis='columns')
MOU1_pos_df.to_csv('pos_MOU1.csv')

# import pos_df with correct POS tags from csv
new_MOU1_pos = pd.read_csv('cor_pos_MOU1.csv')
num_article = len(tree_MOU1)
article_content = []
new_tree_MOU1 = []
article_order = []
for i in range(1, (num_article + 1)):
    article_data = new_MOU1_pos[new_MOU1_pos['Article #'] == i]
    num_sent = article_data['Sentence #'].nunique() 
    for j in range (1, (num_sent + 1)):
        sent_data = article_data[article_data['Sentence #'] == j]
        word_tuple = [tuple(word) for word in sent_data[['Word','POS']].values]
        article_content.append(word_tuple)
        article_order.append(i)
new_tree_MOU1.extend(article_content)

# chunk MOU1 compensation sentences with classifier_chunker
chunked_MOU1 = [classifier_chunker.parse(sent) for sent in new_tree_MOU1]

# deep chunk compensation sentences
deep_rule = """
# Chunk Clauses
CL:     {<NP>?<IN>?<VP>?<NP|NN><VP><NP>?}
        {<NP><WRB><VP><CD><NN><NUM><CC><JJ>}
        {<NP>+<WP><VP><NP>}
# Chunk Num Phrases
NPH:    {<NUM><DT>?<NN><VP>?<TO><VP>}
        {<NUM><VP>?<NP>+}
        {<NUM><NN|NP>?<CD>+}
        {<NUM><IN><NP><IN><NP>}
        {<NUM><CL><RB><CC>?<NP>?}
"""
deep_chunkParser = RegexpParser(deep_rule)
deep_chunk = [deep_chunkParser.parse(i) for i in chunked_MOU1]

# extract CL and NPH chunk to summarize bonus information
bonus_info_df = pd.DataFrame(columns = ['Section', 'Employee', 'Pay'])
for i in range(0, len(deep_chunk)):
    emp_info = [' '.join([l[0] for l in s.leaves()]) for s in deep_chunk[i] if type(s) == Tree and s.label() == 'CL']
    pay_info = [' '.join([l[0] for l in s.leaves()]) for s in deep_chunk[i] if type(s) == Tree and s.label() == 'NPH']
    if (len(emp_info) != 0 and len(pay_info) != 0):
        for info in pay_info:
            art_info = MOU1_article[article_order[i]-1]
            new_entry = {'Section': art_info, 'Employee': emp_info[0], 'Pay': info}
            bonus_info_df.loc[len(bonus_info_df)] = new_entry
        emp = pay = []

# export extracted info to csv file
bonus_info_df.to_csv('bonus_info.csv')

# create sql database from csv file
con = sqlite3.connect('bonus_info.db')
cur = con.cursor()
cur.executescript("""
        DROP TABLE IF EXISTS bonus_table;
        CREATE TABLE bonus_table (Section, Employee, Pay);
        """)
with open('bonus_info.csv','r', encoding='utf8') as fin:
    dr = csv.DictReader(fin)
    to_db = [(i['Section'], i['Employee'], i['Pay']) for i in dr]
cur.executemany("INSERT INTO bonus_table (Section, Employee, Pay) VALUES (?, ?, ?);", to_db)
con.commit()
con.close()

# build CFG structure to translate input question into a SQL query
SQL_rule = """
% start S
S[SEM=(?np + ?vp + ?nn + WHERE + ?pnp)] -> NP[SEM=?np] VP[SEM=?vp] NN[SEM=?nn] PNP[SEM=?pnp]

NP[SEM=(?det + ?nn)] -> DT[SEM=?det] NN[SEM=?nn]
DT[SEM='SELECT'] -> 'What'|'Which'
NN[SEM='Employee,'] -> 'employees'

VP[SEM=(?md + ?vp)] -> MD[SEM=?md] VP[SEM=?vp]
MD[SEM=''] -> 'will'
VP[SEM=(?v + ?np)] -> V[SEM=?v] NP[SEM=?np]
V[SEM=''] -> 'receive'
NP[SEM=(?wp + ?nn + ?in)] -> WP[SEM=?wp] NN[SEM=?nn] IN[SEM=?in]
WP[SEM=''] -> 'what'
NN[SEM='Pay'] -> 'percentage/amount'
IN[SEM=''] -> 'of'

NN[SEM='FROM bonus_table'] -> 'bonus'

PNP[SEM=(?p + ?np)] -> P[SEM=?p] NP[SEM=?np]
P[SEM=''] -> 'under'
NP[SEM=(?jj + ?nn)] -> JJ[SEM=?jj] NN[SEM=?nn]
JJ[SEM='Section="BILINGUAL BONUS"'] -> 'Bilingual'
NN[SEM=''] -> 'condition'

JJ[SEM='Section="OVERTIME"'] -> 'Overtime'
JJ[SEM='Section="SHIFT DIFFERENTIAL"'] -> 'Shift_Differential'
JJ[SEM='Section="SALARIES"'] -> 'Salaries'
JJ[SEM='Section="SIGN LANGUAGE PREMIUM"'] -> 'Sign_Language'
JJ[SEM='Section="COURT APPEARANCES"'] -> 'Court_Appearances'
JJ[SEM='Section="CIVIC DUTY"'] -> 'Civic_Duty'
JJ[SEM='Section="JURY SERVICE"'] -> 'Jury_Service'
JJ[SEM='Section="MILITARY LEAVE"'] -> 'Military_Leave'
JJ[SEM='Section="CALL BACK PAY"'] -> 'Call_Back'
JJ[SEM='Section="DISTURBANCE CALLS"'] -> 'Disturbance_Calls'
JJ[SEM='Section="ON-CALL/STANDBY COMPENSATION"'] -> 'On-Call/Standby'
JJ[SEM='Section="TEMPORARY SUPERVISORY PAY"'] -> 'Temporary_Supervisory'
JJ[SEM='Section="CIVILIAN SUPERVISORY DIFFERENTIAL"'] -> 'Civilian_Supervisory'
JJ[SEM='Section="MILEAGE"'] -> 'Mileage'
"""
SQL_grammar = grammar.FeatureGrammar.fromstring(SQL_rule)
SQL_parser = parse.FeatureEarleyChartParser(SQL_grammar)

# simple application to allow user search bonus info based on conditions
def app():
    condition = ''
    section_names = ['Salaries','Overtime','Shift_Differential','Bilingual','Sign_Language','Court_Appearances','Civic_Duty','Jury_Service','Military_Leave','Call_Back','Disturbance_Calls','On-Call/Standby','Mileage','Temporary_Supervisory']
    while condition not in section_names:
        condition = str(input('\n Salaries, Overtime, Shift_Differential, Bilingual, Sign_Language, Court_Appearances, Civic_Duty, Jury_Service, Military_Leave, Call_Back, Disturbance_Calls, On-Call/Standby, Mileage, Temporary_Supervisory, Civilian_Supervisory.\nPlease type in a correct section you want to search for bonus info: '''))
    question = 'Which employees will receive what percentage/amount of bonus under ' + str(condition) + ' condition'
    
    # translate input question into SQL query with grammar structure
    query_tree = SQL_parser.parse(question.split())
    for tree in query_tree:
        root = tree.label()['SEM']
    SQL_query = list(root)
    query = ' '.join(str(w) for w in SQL_query if w)
    
    # connect to SQL database and execute query to retrieve results
    con = sqlite3.connect('bonus_info.db')
    cur = con.cursor()
    rows = cur.execute(query)
    rows_con = rows.fetchall()
    num_rows = len(rows_con)
    if num_rows == 0:
        print('No bonus info with percentage or amount in this section')
    else:
        for r in rows_con:
            print(r)
    con.commit()
    con.close()

# test the app
app()


