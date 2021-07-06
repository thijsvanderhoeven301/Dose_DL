import numpy as np
import pandas as pd


def max_dose(dose, struct_arr, structure):
    """Calculates max dose within a structure

    :params dose: array with dose information
    :params struct_arr: structure information
    :params structure: name of structure
    :return dmax: Maximum dose
    """
    struc_list = r'C:/Users/thijs/Documents/master applied physics/mep/project_repository/Dose_DL/Lists/Structures.txt'
    with open(struc_list) as s:
        structs = [line.rstrip()[1:-1] for line in s]
    structs.append('PTVpros+vs')
    if structure in structs:
        num = structs.index(structure)
    struct_arr = np.squeeze(struct_arr[num, :, :, :])
    dmax = max(dose[struct_arr])
    return dmax


def mean_dose(dose, struct_arr, structure):
    """Calculates mean dose within a structure

    :params dose: array with dose information
    :params struct_arr: structure information
    :params structure: name of structure
    :return dmean: Mean dose
    """
    struc_list = r'C:\Users\t.meerbothe\Desktop\project_Thierry_64_dev/Lists/Structures.txt'
    with open(struc_list) as s:
        structs = [line.rstrip()[1:-1] for line in s]
    structs.append('PTVpros+vs')
    if structure in structs:
        num = structs.index(structure)
    struct_arr = np.squeeze(struct_arr[num, :, :, :])
    dmean = np.mean(dose[struct_arr])
    return dmean


def Dx_calc(dose, struct_arr, x, structure):
    """Calculates the Dx within a structure. Ex: D95

    :params dose: array with dose information
    :params struct_arr: structure information
    :params x: Percentage of the statistic (between 0 and 1)
    :params structure: name of structure
    :return Dx: Dx characteristic
    """
    struc_list = r'C:\Users\t.meerbothe\Desktop\project_Thierry_64_dev/Lists/Structures.txt'
    with open(struc_list) as s:
        structs = [line.rstrip()[1:-1] for line in s]
    structs.append('PTVpros+vs')
    if structure in structs:
        num = structs.index(structure)
    struct_arr = np.squeeze(struct_arr[num, :, :, :])
    PTV_vox = dose[struct_arr]
    Nvox = PTV_vox.size
    req = Nvox - int(round(x*Nvox))
    PTV_vox = np.reshape(PTV_vox, [Nvox])
    PTV_sort = np.sort(PTV_vox)
    Dx = PTV_sort[req]
    return Dx


def Vx_calc(dose, struct_arr, x, structure):
    """Calculates the Vx within a structure. Ex: V45

    :params dose: array with dose information
    :params struct_arr: structure information
    :params x: Percentage of the statistic (between 0 and 1)
    :params structure: name of structure
    :return Vx: Vx characteristic
    """
    struc_list = r'C:\Users\t.meerbothe\Desktop\project_Thierry_64_dev/Lists/Structures.txt'
    with open(struc_list) as s:
        structs = [line.rstrip()[1:-1] for line in s]
    structs.append('PTVpros+vs')
    if structure in structs:
        num = structs.index(structure)
    struct_arr = np.squeeze(struct_arr[num, :, :, :])
    PTV_vox = dose[struct_arr]
    NVox = PTV_vox.size
    ReqVox = PTV_vox >= x
    Vx = sum(ReqVox)/NVox
    return Vx


def CI_calc(dose, struct_arr, ref_dose):
    """Calculates the conformity index within a structure.

    :params dose: array with dose information
    :params struct_arr: structure information
    :params ref_dose: prescription dose
    :return CI: Conformity index
    """
    PTV_arr = np.squeeze(struct_arr[-1, :, :, :])
    EXT_arr = np.squeeze(struct_arr[3, :, :, :])
    PTV_vox = dose[PTV_arr]
    NPTVvox = PTV_vox.size
    EXT_vox = dose[EXT_arr]
    PTVcon = PTV_vox >= ref_dose
    EXTcon = EXT_vox >= ref_dose
    CI = sum(PTVcon)/NPTVvox*sum(PTVcon)/sum(EXTcon)
    return CI


def HI_calc(dose, struct_arr, D98):
    """Calculates the homogeneity index within a structure.

    :params dose: array with dose information
    :params struct_arr: structure information
    :params D98: D98 statistic
    :return HI: Homogeneity index
    """
    D2 = Dx_calc(dose, struct_arr, 0.02, 'PTVpros+vs')
    D50 = Dx_calc(dose, struct_arr, 0.5, 'PTVpros+vs')
    return (D2 - D98)/D50


def isodose_dice(dose, pr_dose, pct):
    """Calculates the DICE for isodose volumes between actual
    and predicted dose

    :params dose: array with dose true information
    :params pr_dose: array with predicted dose information
    :params pct: percentages at which to calculate
    :return DICE: DICE score
    """
    dose_arr = dose > pct*60
    pr_dose_arr = pr_dose > pct*60
    DICE = 2*np.sum(np.logical_and(dose_arr, pr_dose_arr))/(np.sum(dose_arr) + np.sum(pr_dose_arr))
    return DICE


def DVH_calc(dose, struct_arr, ID):
    """Calculates the DVH of a structure

    :params dose: array with dose information
    :params struct_arr: structure information
    :params ID: name of structure
    :return df: DVH of structure
    """
    struc_list = r'C:\Users\t.meerbothe\Desktop\project_Thierry_64_dev/Lists/Structures.txt'
    with open(struc_list) as s:
        structs = [line.rstrip()[1:-1] for line in s]
    structs.append('PTVpros+vs')
    df = pd.dataframe(columns=['ID', 'structure', 'DVH'])
    for structure in structs:
        num = structs.index(structure)
        struct_arr = np.squeeze(struct_arr[num, :, :, :])
        PTV_vox = dose[struct_arr]
        Nvox = PTV_vox.size
        PTV_vox = np.reshape(PTV_vox, [Nvox])
        PTV_sort = np.sort(PTV_vox)
        df.append({'ID': ID,
                           'structure': structure,
                           'DVH': PTV_sort})
    return df