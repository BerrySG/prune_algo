import numpy as np
import matplotlib.pyplot as plt


# pruneWay: oneShot, iterative
def fc_plt(epochs, times, percents, pruneWay):

    avg_x = np.zeros(shape=(len(percents), epochs*3))
    avg_y = np.zeros(shape=(len(percents), epochs*3))
    for i in range(0, len(percents)):
        x = np.zeros(shape=(times, epochs*3))
        y = np.zeros(shape=(times, epochs*3))
        for j in range(1, times + 1):
            x_data = np.load('test1_Datum/x_' + str(percents[i]) + '_' + pruneWay + '_' + str(j) + '.npy')
            y_data = np.load('test1_Datum/y_' + str(percents[i]) + '_' + pruneWay + '_' + str(j) + '.npy')
            x[j-1] = x_data
            y[j-1] = y_data
        avg_x[i] = np.mean(x, axis=0)
        avg_y[i] = np.mean(y, axis=0)

    for i in range(0, len(percents)):
        if percents[i] >= 1:
            plt.plot(avg_x[i], avg_y[i], label='%.1f'%(percents[i]))
        else:
            plt.plot(avg_x[i], avg_y[i], label='%.1f' % (percents[i]*100))
    plt.axis([0, 300 * epochs, 0.93, 0.98])
    plt.legend(loc='lower right')
    plt.show()

def test_ply(epochs, percents, pruneWay):
    x = np.zeros(shape=(len(percents), 3*epochs))
    y = np.zeros(shape=(len(percents), 3*epochs))
    x[0] = np.load('test_Datum/x_100_oneShot_1.npy')
    y[0] = np.load('test_Datum/y_100_oneShot_1.npy')
    for i in range(1, len(percents)):
        x[i] = np.load('test_Datum/x_' + str(percents[i]) + '_' + pruneWay + '_1.npy')
        y[i] = np.load('test_Datum/y_' + str(percents[i]) + '_' + pruneWay + '_1.npy')


    for i in range(0, len(percents)):
        plt.plot(x[i], y[i], label='%.1f'%(percents[i]))
    plt.axis([0, 300 * epochs, 0.93, 0.98])
    plt.legend(loc='lower right')
    plt.show()




if __name__ == "__main__":
    percents = [100, 0.9, 0.75, 0.351]
    # test_ply(30, percents, 'oneShot')
    fc_plt(60, 5, percents, 'oneShot')


if __name__ == "__main__":
    percents = [100, 0.9, 0.75, 0.351]
    # test_ply(30, percents, 'oneShot', 'no_absolute_Datum/')
    # fcn_plt(60, 5, percents, 'oneShot', '')
    x = np.load('Conv_2_Datum/x_conv2-100_fc-100_iter-0_0.npy')
    y = np.load('Conv_2_Datum/y_conv2-100_fc-100_iter-0_0.npy')
    print(x)

    plt.plot(x, y)
    plt.axis([0, 30000, 0.8, 0.93])
    plt.show()



