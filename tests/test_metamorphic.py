import numpy as np

def test_metamorphic_transformation(model, data):
    test_images, _ = data
    flipped_images = [np.flip(img, axis=1) for img in test_images]
    predictions_original = model.predict(test_images)
    predictions_flipped = model.predict(flipped_images)
    assert np.array_equal(predictions_original, predictions_flipped)
