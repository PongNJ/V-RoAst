import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import random


def gaussian_noise(img):
    mean = 0
    stddev = 180
    noise = np.zeros(img.shape, np.uint8)
    cv2.randn(noise, mean, stddev)
    noisy_img = cv2.add(img, noise)
    return noisy_img

def salt_pepper_noise(img, prob):
    noisy_img = img.copy()
          
    black = np.array([0, 0, 0], dtype='uint8')
    white = np.array([255, 255, 255], dtype='uint8')

    probs = np.random.random(noisy_img.shape[:2])
    noisy_img[probs < (prob / 2)] = black
    noisy_img[probs > 1 - (prob / 2)] = white
    return noisy_img


def speckle_noise(image):
    noise = np.random.randn(*image.shape)
    noisy_image = image + image * noise
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    
    return noisy_image


def periodic_noise(image, frequency, amplitude):
    image = image.astype(np.float32)
    rows, cols = image.shape[:2]
    
    x = np.arange(cols)
    y = amplitude * np.sin(2 * np.pi * frequency * x / cols)
    noise = np.tile(y, (rows, 1))
    
    noisy_image = image + noise[:, :, np.newaxis]
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    
    return noisy_image


def quantization_noise(image, levels):
    step = 256 // levels
    noisy_image = (image // step) * step
    
    return noisy_image


if __name__ == '__main__':
    
    image_file_path = "../traffic_data/ThaiRAP/"
    save_file_path = "../traffic_data/augmented_images/"

    if not os.path.exists(save_file_path):
        os.makedirs(save_file_path)
        
    to_augment = [48, 49, 50, 51, 84, 85, 86, 87, 88, 89, 90, 91, 132, 133, 134, 135, 136, 137, 138, 139, 504, 505, 506, 
                   507, 508, 509, 510, 511, 559, 771, 772, 773, 774, 775, 776, 777, 778, 795, 796, 797, 798, 811, 948, 949, 
                   950, 951, 1413, 1414, 1415, 1416, 1816, 1817, 1818, 1819, 112, 113, 114, 115]
    
    image_files = [str(f) + '.jpg' for f in to_augment]
    functions = [gaussian_noise, salt_pepper_noise, speckle_noise, periodic_noise, quantization_noise]
        
    for file in image_files:
        for i in range(len(functions)):
            # print('------------------------------------')
            image_path = os.path.join(image_file_path, file)
            img = cv2.imread(image_path, 1)

            if i == 1:
                noisy_img = functions[i](img, 0.05)
                cv2.imwrite(os.path.join(save_file_path, 'augmented_' + str(i+1) + '_' + str(1) + '_' + file), noisy_img)
                noisy_img = functions[i](img, 0.1)
                cv2.imwrite(os.path.join(save_file_path, 'augmented_' + str(i+1) + '_' + str(2) + '_'+ file), noisy_img)
                
            elif i == 3:
                noisy_img = functions[i](img, 5, 50)
                cv2.imwrite(os.path.join(save_file_path, 'augmented_' + str(i+1) + '_' + str(1) + '_' + file), noisy_img)
                noisy_img = functions[i](img, 10, 70)
                cv2.imwrite(os.path.join(save_file_path, 'augmented_' + str(i+1) + '_' + str(2) + '_'+ file), noisy_img)
            
            elif i == 4:
                noisy_img = functions[i](img, 8)
                cv2.imwrite(os.path.join(save_file_path, 'augmented_' + str(i+1) + '_' + str(1) + '_' + file), noisy_img)
                noisy_img = functions[i](img, 16)
                cv2.imwrite(os.path.join(save_file_path, 'augmented_' + str(i+1) + '_' + str(2) + '_'+ file), noisy_img)
                
            else:
                noisy_img = functions[i](img)
                cv2.imwrite(os.path.join(save_file_path, 'augmented_' + str(i+1) + '_' + file), noisy_img)
                
    print(f'The number of images to be augmented: {len(image_files)} and the number of augemented images: {len(os.listdir(save_file_path))}')