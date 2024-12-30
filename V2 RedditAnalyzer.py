#For Roberta
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax

#For NLP
import re

#For Data Collection
import praw
from tqdm import tqdm
from datetime import datetime



def chunkstring(string, length):
    return (string[0+i:length+i] for i in range(0, len(string), length))

class RedditRoberta:

    def __init__(self, MODEL = "cardiffnlp/twitter-roberta-base-sentiment"):
            
            self.reddit = praw.Reddit(
                client_id = "271r6fj47QNd_qLQp7W5bg",
                client_secret = "-fi243nTTrDfSs5c_shNFnBO7OgUfw",
                user_agent = "Stock Board Crawler")
            
            self.MODEL = MODEL
            self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.MODEL)
            
    def analyze(self, snippet):
        encoded_text = self.tokenizer(snippet, return_tensors='pt')
        output = self.model(**encoded_text)
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)
        return {
            'roberta_neg': scores[0],
            'roberta_neu': scores[1],
            'roberta_pos': scores[2],
        }

    def search_board_for(self, board, keyword, hours_ago=None):

        self.keyword = keyword
        self.board = board

        def time_comparison(submission_timestamp, hours_ago):
            after_this_date = datetime.now().timestamp() - (3600 * hours_ago)
            if after_this_date - submission_timestamp <= 0:
                return True
            else:
                return False

        search_paramaters = {'query':keyword.lower(), 'limit':None}
        self.searched_data = self.reddit.subreddit(board).search(**search_paramaters)

        self.submission_body = []
        for submission in self.searched_data:
            if not hours_ago:
                if keyword in submission.title:
                    post_text = submission.selftext.lower()
                    post_text += submission.title
                    post_text = re.sub(r"\n", " ", post_text)
                    #post_text = re.sub(r"\S*http\S*", "", post_text)
                    if len(post_text) >= 514:
                        list = chunkstring(post_text, 514)
                        for item in list:
                            self.submission_body.append(item)
                    if len(post_text) < 514:
                        self.submission_body.append(post_text)
            else:
                if time_comparison(submission.created_utc, hours_ago):
                    if keyword in submission.title:
                        post_text = submission.selftext.lower()
                        post_text += submission.title
                        post_text = re.sub(r"\n", " ", post_text)
                        #post_text = re.sub(r"\S*http\S*", "", post_text)
                        if len(post_text) >= 514:
                            list = chunkstring(post_text, 514)
                            for item in list:
                                self.submission_body.append(item)
                        if len(post_text) < 514:
                            self.submission_body.append(post_text)
                        
        return self.submission_body
    
    def analyze_chunks(self):
        total_score = {"neg": 0, "neu": 0, "pos": 0}
        for submission in tqdm(self.submission_body, desc="Robertaing"):
            output = self.analyze(submission)
            total_score["neg"] += output['roberta_neg']
            total_score["neu"] += output['roberta_neu']
            total_score["pos"] += output['roberta_pos']        
        for key in total_score:
            try:
                total_score[key] /= len(self.submission_body)
            except ZeroDivisionError as e:
                return print("There were no posts on the specified subbreddit with the specified keyword")
                
        return print(f"The Keyword ({self.keyword}) on the subreddit ({self.board}) is {total_score}"), total_score

if __name__ == "__main__":
    analyze = RedditRoberta()
    analyze.search_board_for("stocks", "Trump")
    analyze.analyze_chunks()
