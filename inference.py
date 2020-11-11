import os as os
import numpy as np
from model import get_model
from keras.optimizers import Adam
import json

test_X = []

test_dir = './Complete_dataset/image_with_semantics_objs'

image_names = []
for img_name in os.listdir(test_dir)[4000:]:
    img = np.load(os.path.join(test_dir, img_name))
    test_X.append(img)
    image_names.append(img_name)

test_X = np.array(test_X)

opt = Adam(0.001)
model = get_model(channels=10)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
model.load_weights('./Models/saved-model-100-0.94.hdf5')

preds = model.predict(test_X, verbose=True)
preds = np.argmax(preds, axis=1)

preds_dict = {}
for i in range(len(image_names)):
    preds_dict[image_names[i]] = 'Place_'+str(int(preds[i]))

with open('./preds4.json', 'w') as f:
    json.dump(preds_dict, f, indent=1)
