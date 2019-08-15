def load_dataset():
    num = random.randint(0, 30)
    train_dset = h5py.File('train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dset['train_set_x'][:])
    train_set_y_orig = np.array(train_dset['train_set_y'][:])
    test_dset = h5py.File('train_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dset['train_set_x'][:])
    test_set_y_orig = np.array(test_dset['train_set_y'][:])
    classes = np.array(test_dset['list_classes'][:])
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))  # always pass in a tuple
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))  # always pass in a tuple
    print(train_set_x_orig[num][num][num], test_set_x_orig[num][num][num])
    print(train_set_x_orig.shape, test_set_x_orig.shape)
return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


# %%
# w, b, X, Y = np.array([[1], [2]]), 2, np.array([[1, 2], [3, 4]]), np.array([[1, 0]])
# print(propagate(w, b, X, Y))


# # def print_image(index):
# #%%
# index = 25
# train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()
# plt.imshow(train_set_x_orig[5])
# print("y = " + str(train_set_y[:, index]) + ", it's a '" +
#     classes[np.squeeze(train_set_y[:, index])].decode("utf-8") + "' picture.")


# print(np.squeeze(train_set_x_orig, axis=1).shape)
# w, b, X, Y = np.array([[1.], [2.]]), 2., np.array(
#     [[1., 2., -1.], [3., 4., -3.2]]), np.array([[1, 0, 1]])
# print(train_set_x_orig.shape[0])
# print(w.shape)
# flat_x = flatten_data(train_set_x_orig)
# flat_y = flatten_data(train_set_y)
# print(propagate(w, b, flat_x, train_set_y))


# w, b = zero_param_init(standard_train_x.shape[0])
# grads, cost = propagate(w, b, standard_train_x, train_set_y)
# w, b, X, Y = np.array([[1], [2]]), 2, np.array(
#     [[1, 2], [3, 4]]), np.array([[1, 0]])
# grads, cost = propagate(w, b, X, Y)
# print(grads, cost)
