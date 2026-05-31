import json

with open("bdd100k_labels_images_train.json", "r") as f:
    data = json.load(f)

print("First 5 image names:")
for item in data[:5]:
    print(item.get("name"))
