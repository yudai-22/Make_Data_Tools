import numpy as np
from astropy.io import fits
from cut_resize_tools import (
    slide, remove_nan, parallel_processing, process_data_segment,
    normalization, select_conv, proccess_npyAppend, maximum_value_determination, normalization_sigma
)
from tqdm import tqdm
from Make_Data_Tools import resize

# Constants
VSmooth = 5
Thresh = 1
Sigma = 1
Sch_RMS = 10
Ech_RMS = 90
Sch_II = 121
Ech_II = 241
Percentile = 99.998

Cut_size_list = [356, 156, 84, 36]
Integrate_layer_num = 30
Obj_size = 100
Obj_sig = 7.5
FITS_PATH = "/home/filament/fujimoto/fits/Cygnus_sp16_vs-40_ve040_dv0.25_12CO_Tmb.fits"
OUTPUT_DIR = "/home/filament/fujimoto/Cygnus-X_CAE/data/zroing_resize_data/resize_data/sigma/"

def process_fits_data(fits_path, cut_size_list, sch_ii, ech_ii, vsmooth, thresh, sigma, sch_rms, ech_rms, integrate_layer_num, percentile, output_dir, obj_size, obj_sig):
    # Load FITS file
    with fits.open(fits_path, memmap=True) as hdul:
        raw_data = hdul[0].data
        # header = hdul[0].header # Not used

        max_thresh = maximum_value_determination(
            "sigma", raw_data, sch_ii, vsmooth, ech_ii, sch_rms, ech_rms,
            sigma, thresh, integrate_layer_num, obj_size, obj_sig, percentile
        )

        for pix in cut_size_list:
            print(f"Processing data clipped to {pix} pixels...")

            # Step1: スライス (不要な変数格納は避ける)
            cut_data = slide(raw_data[sch_ii:ech_ii], pix+4)
            print(f"Number of data clipped to {pix} pixels: {len(cut_data)}")

            cut_data = remove_nan(cut_data)
            print(f"Number of data after deletion: {len(cut_data)}")

            # Step2: 並列前処理
            processed_list = parallel_processing(
                process_data_segment, cut_data,
                sigma=sigma, vsmooth=vsmooth, thresh=thresh, sch_rms=sch_rms, ech_rms=ech_rms,
                integrate_layer_num=integrate_layer_num
            )
            del cut_data

            print("Start convolution, resizing and changing max value")
            conv_resize_list = []
            for _data in tqdm(processed_list):
                _data = select_conv(_data, obj_size, obj_sig)
                _data = resize(_data, (obj_size, obj_size))
                # _data = np.clip(_data, a_min=None, a_max=max_thresh)
                conv_resize_list.append(_data)
            del processed_list

            
            conv_resize_list = normalization_sigma(conv_resize_list, sigma=max_thresh, multiply=15)
            output_file = f"{output_dir}CygnusX_cut_sigma_resize_to_{obj_size}x{obj_size}"
            proccess_npyAppend(output_file, conv_resize_list)
            del conv_resize_list
            print(f"Data saved to {output_file}")

def main():
    process_fits_data(
        fits_path=FITS_PATH,
        cut_size_list=Cut_size_list,
        sch_ii=Sch_II,
        ech_ii=Ech_II,
        vsmooth=VSmooth,
        thresh=Thresh,
        sigma=Sigma,
        sch_rms=Sch_RMS,
        ech_rms=Ech_RMS,
        obj_size=Obj_size,
        obj_sig=Obj_sig,
        output_dir=OUTPUT_DIR,
        integrate_layer_num=Integrate_layer_num,
        percentile=Percentile
    )

if __name__ == "__main__":
    main()
