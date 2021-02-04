from tensorflow import GradientTape, constant
import tensorflow.keras.losses as losses
import numpy as np
import argparse

from resources import models, data, storage, visualize


def fgsm(x, y, model, loss=losses.CategoricalCrossentropy(), eta=0.1):
    """Perform an untargeted fast gradient method attack
    Arguments:
        x {np.ndarray} -- target sample
        y {np.ndarray} -- true label
        model {tf.keras.Model} -- target model
        loss {tf.keras.losses.Loss} -- target loss function
    Keyword Arguments:
        eta {float} -- step size for FGM (default: {0.1})
    """
    # convert x to tf.Tensor
    x_t = constant(x)
    model.trainable = False

    with GradientTape() as tape:
        # explicitly add input tensor to tape
        tape.watch(x_t)
        # get prediction
        pred = model(x_t)
        # calculate loss
        l = loss(y, pred)

    # calculate dloss/dx
    gradients = tape.gradient(l, x_t)
    # visualize.gradients(x,y,pred,gradients)

    # if we are still classifying correctly, pertubate normally
    # if we already crossed the decision boundary, go back to original class
    predicted_label = np.argmax(pred)
    correct_label = np.argmax(y)
    correctness_sign = 1 if predicted_label == correct_label else -1

    x_adv = x + eta * np.sign(gradients) * correctness_sign
    # clip x_adv to valid range
    x_adv = np.clip(x_adv, 0, 1)
    return x_adv


def eta(iteration: int, l=1.2, initial=0.1):
    """Generate exponentially decaying step size based on current iteration

    Args:
        iteration (int): the current iteration
        l (float, optional): lambda for exponential decay. Defaults to 1.2.
        initial (float, optional): initial value for exponential decay. Defaults to 0.1.

    Returns:
        float: eta for FGSM
    """
    return initial * np.exp(-l * iteration)


def repeated_fgsm(model, x, y, max_iterations, c):
    original_x = x
    true_label = np.argmax(y.flatten())
    for i in range(max_iterations):
        prediction = model(x).numpy()
        sorted_prediction = np.sort(prediction.flatten())
        confidence_diff = sorted_prediction[-1] - sorted_prediction[-2]
        predicted_class = np.argmax(prediction.flatten())
        if confidence_diff < 1e-7 and predicted_class != true_label:
            break

        eta = 1e-9 + c * confidence_diff
        x = fgsm(x, y, model, eta=eta)

    print(f"Generation took {i} iterations")
    prediction = model(x)
    sorted_prediction = np.sort(prediction.numpy().flatten())
    print("Sorted Prediction:")
    print(sorted_prediction)
    print(f"Confidence diff: {sorted_prediction[-1] - sorted_prediction[-2]}")
    image_diff = original_x - x
    print(f"Image diff: min={np.min(image_diff)}, max={np.max(image_diff)}, mean={np.mean(image_diff)}")
    return x, prediction


def generate_single(model_type: str, max_iterations: int, c: float, visualize_result: bool):
    model_type = model_type.lower()
    if model_type == 'mnist':
        model = models.build_mnist_cnn()
        x, y = data.get_single_mnist_test_sample(include_label=True)
    elif model_type == 'fmnist':
        model = models.build_fmnist()
        x, y = data.get_single_fmnist_test_sample(include_label=True)
    elif model_type == 'single_layer':
        model = models.build_single_layer()
        x, y = data.get_single_mnist_test_sample(include_label=True)
    elif model_type == 'single_conv_layer':
        model = models.build_single_layer()
        x, y = data.get_single_mnist_test_sample(include_label=True)
    elif model_type == 'cifar10':
        model = models.build_cifar_10()
        x, y = data.get_single_cifar_10_test_sample(include_label=True)
    else:
        raise SystemError(f"Unknown model type {model_type}")

    storage.load_weights(model)

    x_adv, pred_adv = repeated_fgsm(model, x, y, max_iterations, c)

    pred_adv = pred_adv.numpy().flatten()
    print(f"Predicted class: {np.argmax(pred_adv)}")
    print(f"True label: {np.argmax(y)}")

    # visualize prediction
    if visualize_result:
        visualize.prediction(x, y, pred_adv)
    response = input("Save generated boundary sample? (y/N)")
    if response.lower() == 'y':
        print("Saving...")
        np.save(f"boundarys/{model_type}.npy", x_adv)
    else:
        print("Not saving")


def generate_batch(model_type: str, num_images: int, max_iterations: int, c: float):
    model_type == model_type.lower()
    if model_type == 'mnist':
        model = models.build_mnist_cnn()
        x_test, y_test = data.get_mnist_data(test_only=True)
    elif model_type == 'fmnist':
        model = models.build_fmnist()
        x_test, y_test = data.get_fmnist_data(test_only=True)
    elif model_type == 'single_layer':
        model = models.build_single_layer()
        x_test, y_test = data.get_mnist_data(test_only=True)
    elif model_type == 'single_conv_layer':
        model = models.build_single_layer()
        x_test, y_test = data.get_mnist_data(test_only=True)
    elif model_type == 'cifar10':
        model = models.build_cifar_10()
        x_test, y_test = data.get_cifar_10_data(test_only=True)
    else:
        raise SystemError(f"Unknown model type {model_type}")

    storage.load_weights(model)
    x_test = x_test[:num_images]
    y_test = y_test[:num_images]

    x_adv = np.empty_like(x_test)
    pred_adv = np.empty((x_test.shape[0], 10))
    for i in range(x_test.shape[0]):
        x = x_test[i:i + 1]
        y = y_test[i:i + 1]

        current_x, current_pred = repeated_fgsm(model, x, y, max_iterations, c)
        x_adv[i] = current_x
        pred_adv[i] = current_pred

    np.save(f"boundaries/{model_type}_batch.npy", x_adv)
    sorted_predictions = np.sort(pred_adv, axis=1)
    diffs = sorted_predictions[:, -1] - sorted_predictions[:, -2]

    print(f"Number of incorrect labels: {np.sum(pred_adv == y_test)}")
    print(
        f"Confidence stats: min={np.min(diffs)}, max={np.max(diffs)}, mean={np.mean(diffs)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Generate a sample that is very close to the decision boundary")
    parser.add_argument("model_type", type=str,
                        help="One of {mnist, mnist_mlp, single_layer, single_conv_layer, cifar10, fmnist}. The model to generate for.")
    parser.add_argument("-i --iterations", dest="iterations",
                        type=int, default=300, help="The maximum number of iterations")
    parser.add_argument("-c", type=float,
                        default=0.01, help="Constant factor for deriving step size from confidence diff")
    parser.add_argument("--visualize", action="store_true",
                        default=False, help="Visualize generated image")
    parser.add_argument("--batch", metavar='n', type=int, default=None,
                        help="Generate a batch of n boundary samples")

    args = parser.parse_args()
    if args.batch is not None:
        generate_batch(args.model_type, args.batch,
                       args.iterations, args.c)
    else:
        generate_single(args.model_type, args.iterations,
                        args.c, args.visualize)
