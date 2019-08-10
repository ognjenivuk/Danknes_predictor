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
        meme_id = random.choice(memes)
        meme = db_json[meme_id]
        meme_image = cv2.imread(os.path.join(path_to_images, meme['filename'].replace('.jpg', '.png')))
        meme_image = cv2.cvtColor(meme_image, cv2.COLOR_BGR2RGB)
        meme_score = meme['ups']

        yield {
            'image': meme_image,
            'score': math.log(meme_score + 1)
        }
        
def get_train_test(path_to_db_json, path_to_images):
    return {
        'learning': get_random_memes(path_to_db_json, path_to_images, count=20), 
        'test': get_random_memes(path_to_db_json, path_to_images, count=10)
    }

def main():
    images_path = './Data/new_data/merged'
    json_path = './Data/new_data/db.json'
    memes = get_train_test(json_path, images_path)
    for meme in memes['learning']:
        plt.imshow(meme['image'])
        plt.title(meme['score'])
        plt.show()

    guessed, real = [], []
    
    for meme in memes['test']:
        plt.imshow(meme['image'])
        plt.title(' xe xe xD ')
        plt.show()
        gess_score = float(input('score: '))
        real_score = meme['score']
        guessed.append(gess_score)
        real.append(real_score)
        print('real was', real_score)

    plt.plot(guessed, guessed, 'r-')
    plt.plot(guessed, real, 'b.')
    plt.show()
        
if __name__ == "__main__":
    main()    