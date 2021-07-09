import matplotlib.pyplot as plt
import numpy as np


def loss_plot(training_loss, validation_loss, std_train, std_val):
    fig, ax = plt.subplots()
    epochs = len(training_loss)
    ax.plot(np.linspace(0, epochs, epochs), training_loss)
    ax.fill_between(np.linspace(0, epochs, epochs), training_loss - std_train, training_loss + std_train, alpha=0.3)
    ax.plot(np.linspace(0, epochs, epochs), validation_loss)
    ax.fill_between(np.linspace(0, epochs, epochs), validation_loss - std_val, validation_loss + std_val, alpha=0.3)
    plt.title('Training and validation loss')
    plt.legend(['Training_loss', 'Validation_loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()


def dose_diff_plot(output, dose, structure, axis: int = 0):
    diff = output - dose
    fig, ax = plt.subplots()
    if axis == 0:
        plot = ax.imshow(diff[72, :, :], cmap='bwr', vmax=10, vmin=-10)
        ax.contour(structure[72, :, :], levels=1, colors='green')
    elif axis == 1:
        plot = ax.imshow(diff[:, 32, :], cmap='bwr', vmax=10, vmin=-10)
        ax.contour(structure[:, 32, :], levels=1, colors='green')
    elif axis == 2:
        plot = ax.imshow(diff[:, :, 60].T, cmap='bwr', vmax=10, vmin=-10)
        ax.contour(structure[:, :, 60].T, levels=1, colors='green')
    fig.colorbar(plot)
    plt.show()


def DVH_plot(dose, struct_arr, structure, color, style):
    struc_list = r'C:/Users/t.vd.hoeven/Dose_DL/Lists/Structures.txt'
    with open(struc_list) as s:
        structs = [line.rstrip()[1:-1] for line in s]
    structs.append('PTVpros+vs')
    if structure in structs:
        num = structs.index(structure)
    struct_arr = np.squeeze(struct_arr[num, :, :, :])
    PTV_vox = dose[struct_arr]
    Nvox = PTV_vox.size
    PTV_vox = np.reshape(PTV_vox, [Nvox])
    PTV_sort = np.sort(PTV_vox)
    vol = 1 - (np.linspace(0, Nvox-1, Nvox)/Nvox)
    plt.plot(PTV_sort, vol, color, linestyle=style)
    plt.xlabel('dose [Gy]')
    plt.ylabel('Volume')


def DICE_plot(DICE, Dice_step):
    ave_dice = np.average(DICE, axis=0)
    std_dice = np.std(DICE, axis=0)
    #fig, ax = plt.subplots()
    plt.plot(Dice_step, ave_dice)
    plt.fill_between(Dice_step, ave_dice - std_dice, ave_dice + std_dice, alpha=0.3)
    #ax.set_ylim(0, 1)
    plt.xlabel('isodose [%]')
    plt.ylabel('DICE []')
    #plt.title('DICE coefficient for isodose volumes')
    plt.show()


def Contour_plot():
    sw_struct = np.swapaxes(structure, 2, 3)
    test = np.zeros([144, 96, 64])
    test[sw_struct[3]] = 1
    test[sw_struct[-1]] = 2

    colmap = mcolors.ListedColormap(np.random.random((3, 3)))
    colmap.colors[0] = mcolors.ColorConverter.to_rgb(mcolors.CSS4_COLORS['darkblue'])
    colmap.colors[1] = mcolors.ColorConverter.to_rgb(mcolors.CSS4_COLORS['royalblue'])
    colmap.colors[2] = mcolors.ColorConverter.to_rgb(mcolors.CSS4_COLORS['yellow'])

    im = plt.imshow(test[:, 58, :].T, cmap=colmap)
    struct_lab = ['Exterior', 'Body', 'PTV']
    patches = [mpatches.Patch(color=colmap.colors[i], label=struct_lab[i]) for i in range(len(colmap.colors))]
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.axis('off')
    plt.show()

    sw_struct = np.swapaxes(structure, 2, 3)
    test = np.zeros([144, 96, 64])
    test[sw_struct[3]] = 1
    test[sw_struct[0]] = 2
    test[sw_struct[-1]] = 3

    colmap = mcolors.ListedColormap(np.random.random((4, 3)))
    colmap.colors[0] = mcolors.ColorConverter.to_rgb(mcolors.CSS4_COLORS['darkblue'])
    colmap.colors[1] = mcolors.ColorConverter.to_rgb(mcolors.CSS4_COLORS['royalblue'])
    colmap.colors[2] = mcolors.ColorConverter.to_rgb(mcolors.CSS4_COLORS['red'])
    colmap.colors[3] = mcolors.ColorConverter.to_rgb(mcolors.CSS4_COLORS['yellow'])

    im = plt.imshow(test[:, :, 32].T, cmap=colmap)
    struct_lab = ['Exterior', 'Body', 'Rectum', 'PTV']
    patches = [mpatches.Patch(color=colmap.colors[i], label=struct_lab[i]) for i in range(len(colmap.colors))]
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.axis('off')
    plt.show()