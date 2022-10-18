import glob, os
import SimpleITK as sitk
from tqdm import tqdm
import pandas as pd


def compute_dice(mask1, mask2):
    mask1 = mask1!=0
    mask2 = mask2!=0
    overlap = mask1 * mask2
    if mask1.sum()+mask2.sum() == 0:
        return None
    return 2*overlap.sum()/(mask1.sum()+mask2.sum())


if __name__ == '__main__':
    gt_dir = ''
    predict_dir = ''
    predict_lesion_paths = glob.glob(os.path.join(predict_dir, '*_img.nii_lesion.nii.gz'))

    df_dice = pd.DataFrame(columns=['seriesuid', 'dice'])
    for predict_lesion_path in tqdm(predict_lesion_paths):
        sid = os.path.basename(predict_lesion_path).replace('_img.nii_lesion.nii.gz', '')
        gt_path = os.path.join(gt_dir, sid+'_lesion_resize.nii.gz')

        predict_sitk = sitk.ReadImage(predict_lesion_path)
        predict_arr = sitk.GetArrayFromImage(predict_sitk)

        gt_sitk = sitk.ReadImage(gt_path)
        gt_arr = sitk.GetArrayFromImage(gt_sitk)

        dice = compute_dice(gt_arr, predict_arr)
        if dice is None:
            continue

        df_dice.loc[len(df_dice)] = {
            'seriesuid': sid,
            'dice': dice
        }

    df_dice.to_csv(os.path.basename(os.path.dirname(predict_dir))+'.csv')



