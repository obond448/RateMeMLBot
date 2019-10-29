import praw
import os
import urllib
from faceDetectionRater import *
import cloudinary
import imageupload
import re

def rate_img(img):
    imgs = findFace(img)
    
    if imgs != None:
        rating_list = []
        for i in imgs.keys():
            rating_list.append(run(imgs[i]))
        return rating_list
    else:
        return None


def determine_gender(title):
    pattern = re.compile("([0-9][0-9])[mMFf]")
    pattern2 = re.compile("[mMFf]([0-9][0-9])")
    m = pattern.search(title)
    m2 = pattern2.search(title)
    gender = ""
    if m:
        #print(m.group(0), submission.id)
        gender = m.group(0)[2].lower()
    if m2:
        #print(m2.group(0), submission.id)
        gender = m2.group(0)[0].lower()
    return gender


reddit = praw.Reddit("bot1",  user_agent='rater user agent')
sub = reddit.subreddit("RateMe")

MIN_SCORE = 3

#get history of post replies
if not os.path.isfile("posts_replied_to.txt"):
    posts_replied_to = []
else:
    with open("posts_replied_to.txt", "r") as f:
       posts_replied_to = f.read()
       posts_replied_to = posts_replied_to.split("\n")
       posts_replied_to = list(filter(None, posts_replied_to))
       
#image upload setup
cloudinary.config(
      cloud_name = 'dsvvzsmvk',  
      api_key = '821254938148261',  
      api_secret = 'irMNAC-QjZN5qIRUukTUisshyuI'  
    )
tag = "rater"

for submission in sub.top(limit=25, time_filter = "week"): #time_filter = "week",
    
    if submission.score < MIN_SCORE or submission.id in posts_replied_to:
        continue #skip
        
    url = submission.url
    subid = submission.id
    print(submission.title)
    # if (determine_gender(submission.title) != "m"):
    #     print("SKIP GENDER")
    #     continue
    
    
    #print(url[0:18])
    if (url[0:18] == "https://i.redd.it/"):
        urllib.request.urlretrieve(url, "{}.jpg".format(subid))#saves image
        print(subid)
        
        rate_faces_save("{}.jpg".format(subid), "rated-{}.jpg".format(subid))
        
        if (os.path.exists("rated-{}.jpg".format(subid))):
            imageupload.upload_files("rated-{}.jpg".format(subid), tag)
        
        
        
        
        #print(rate_img("{}.jpg".format(subid)))
        #ratings = rate_img("{}.jpg".format(subid))-
        
        # if ratings == None:
        #     continue
        
        # confidence_positive = [ratings[i][0] for i in range(len(ratings))] #confidence they are attractive
        # final_confidence = max(confidence_positive)
        
        # out = round(final_confidence,2)	
        #submission.reply("{:.1f}/10".format(out*10))-
        #posts_replied_to.append(submission.id)-
    else:
        print("SKIP IMAGE")
        
        
# with open("posts_replied_to.txt", "w") as f:
#     for post_id in posts_replied_to:
#         f.write(post_id + "\n")