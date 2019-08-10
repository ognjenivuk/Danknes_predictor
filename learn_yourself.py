#!/usr/bin/env python3
from matplotlib import pyplot as plt
import cv2
import os
import random
import json
import math

def get_random_memes(path_to_db_json, path_to_images, count):
    
    with open(path_to_db_json) as f:
        db_json = json.load(f)
        
    memes = list(db_json.keys())

    for _ in range(count):
        try:
            meme_id = random.choice(memes)
            meme = db_json[meme_id]
            meme_image = cv2.imread(os.path.join(path_to_images, meme['filename'].replace('.jpg', '.png')))
            meme_image = cv2.cvtColor(meme_image, cv2.COLOR_BGR2RGB)
            meme_score = meme['ups']

            yield {
                'image': meme_image,
                'score': math.log(meme_score + 1)
            }
        except:
            pass
        
def get_train_test(path_to_db_json, path_to_images):
    return {
        'learning': get_random_memes(path_to_db_json, path_to_images, count=20), 
        'test': get_random_memes(path_to_db_json, path_to_images, count=20)
    }

def main():
    images_path = './Data/new_data/merged'
    json_path = './Data/new_data/db.json'
    memes = get_train_test(json_path, images_path)
    # for meme in memes['learning']:
    #     plt.imshow(meme['image'])
    #     plt.title(meme['score'])
    #     plt.show()

    guesses = 0
    correct = 0

    for meme_pair in zip(memes['learning'], memes['test']):
        plt.subplot(1, 2, 1)
        plt.imshow(meme_pair[0]['image'])
        plt.subplot(1, 2, 2)
        plt.imshow(meme_pair[1]['image'])
        plt.show()
        guess = int(input('what is better: '))
        print(f'1: {meme_pair[0]["score"]}')
        print(f'2: {meme_pair[1]["score"]}')

        weight = (meme_pair[0]["score"] - meme_pair[1]["score"])**2

        guesses += weight

        if meme_pair[0]['score'] > meme_pair[1]['score']:
            correct += weight if guess == 1 else 0
        else:
            correct += weight if guess == 2 else 0
    
    print(f'accuracy = {correct/guesses : .2f}')
        
if __name__ == "__main__":
    main()    