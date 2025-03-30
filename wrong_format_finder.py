from pathlib import Path
import filetype

# RFC image file extensions supported by TensorFlow
img_exts = {"png", "jpg", "gif", "bmp"}

path = Path("C:\\Users\\peter\\Documents\\SJSU\\CMPE_257_ML\\ML_SJSU_Fall_2024_Grp3\\Bananas")

for file in path.iterdir():
    if file.is_dir():
        continue

    ext = filetype.guess_extension(file)

    if ext is None:
        print(f"'{file}': extension cannot be guessed from content")
    elif ext not in img_exts:
        print(f"'{file}': not a supported image file")