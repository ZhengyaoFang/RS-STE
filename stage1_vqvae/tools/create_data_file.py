import random
import os


def create_data_file(directory):
    image_list = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_list.append(os.path.abspath(os.path.join(root, file)))
    return image_list

    


if __name__ == "__main__":
    root_dirs = [
        "/data/fzy/data/STE2/Union14M",
        "/data/fzy/data/STE2/Tamper",
        "/data/fzy/data/STE/SelfSynTamper",
    ]
    train_file_path = "data/train_Tamper_SelfTamper_Union14M.txt"
    test_file_path = "data/test_Tamper_SelfTamper_Union14M.txt"
    test_num = 1000
    if not os.path.exists("data"):
        os.mkdir("data")

    random.seed(42)
    images = []
    for root_dir in root_dirs:
        images.extend(create_data_file(root_dir))
    random.shuffle(images)
    test_images = images[:test_num]
    train_images = images[test_num:]

    with open(train_file_path, 'w') as f:
        for image in train_images:
            f.write(image + '\n')
    with open(test_file_path, 'w') as f:
        for image in test_images:
            f.write(image + '\n')
    