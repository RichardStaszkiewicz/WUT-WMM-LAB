import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class tools(object):
    def __init__(self) -> None:
        pass

    def readimg(self, dir): # Przeczytaj obrazek z zadanej ścieżki
        image = cv2.imread(dir, cv2.IMREAD_UNCHANGED)
        print(f"image, wymiary: {image.shape}, typ danych:" +
            f"{image.dtype}, wartości: {image.min()} - {image.max()}")
        return image

    def calcPSNR(self, img1, img2): # Oblicz PSNR dla zadanych obrazków
        imax = 255.**2
        mse = ((img1.astype(np.float64)-img2)**2).sum()/img1.size
        return 10.0*np.log10(imax/mse)

    def GaussFiltration(self, nimage, mask_size, dir): # Zastosuj Gaussa
        blurred = cv2.GaussianBlur(nimage, (mask_size, mask_size), 0)
        cv2.imwrite(dir, blurred) # Zapisz obraz we wskazanej ścieżce
        return self.t.calcPSNR(self.org, blurred) # Zwróć PSNR

    def MedianFiltration(self, nimage, mask_size, dir): # Zastosuj Medianę
        blurred = cv2.medianBlur(nimage, mask_size)
        cv2.imwrite(dir, blurred) # Zapisz obraz we wskazanej ścieżce
        return self.t.calcPSNR(self.org, blurred) # Zwróć PSNR


class zad1(tools):
    def __init__(self, org, salt, gauss) -> None: # Zadane obrazy
        self.org = org
        self.salt = salt
        self.gauss = gauss
        self.t = tools()
        self.psnr = [['Filtration', 'Mask', 'Noise', 'PSNR']]

    def executeGauss(self, masks, dir="charts/filtration/Gauss/"):
        for mask in masks: # dla każdej z masek
            p = self.GaussFiltration(self.salt, mask, f"{dir}salt{mask}.png")
            self.psnr.append(['Gauss', mask, "Salt'n'Pepper", p])
            p = self.GaussFiltration(self.gauss, mask, f"{dir}gauss{mask}.png")
            self.psnr.append(['Gauss', mask, 'Gauss', p])

    def executeMedian(self, masks, dir="charts/filtration/Median/"):
        for mask in masks: # dla każdej z masek
            p = self.MedianFiltration(self.salt, mask, f"{dir}salt{mask}.png")
            self.psnr.append(['Median', mask, "Salt'n'Pepper", p])
            p = self.MedianFiltration(self.gauss, mask, f"{dir}gauss{mask}.png")
            self.psnr.append(['Median', mask, 'Gauss', p])


class zad2(tools):
    def __init__(self, org) -> None:
        self.org = org

    def exe(self):
        ans = []
        org_hist = cv2.calcHist([self.org], [0], None, [256], [0, 256])
        ans.append(org_hist.flatten()) # zapisanie oryginalnego histogramu

        col_ycrcb = cv2.cvtColor(self.org, cv2.COLOR_BGR2YCrCb) # zamiana koloru
        col_ycrcb[:, :, 0] = cv2.equalizeHist(col_ycrcb[:, :, 0]) # zm. jasności
        col_val = cv2.cvtColor(col_ycrcb, cv2.COLOR_YCrCb2BGR) # powtót do koloru
        cv2.imwrite("charts/hist/modified.png", col_val) # zapis do pliku

        org_hist = cv2.calcHist([col_val], [0], None, [256], [0, 256])
        ans.append(org_hist.flatten()) # zapisanie zmienionego histogramu
        return ans # Zwrot histogramu (obsługa w osobnym module)


class zad3(tools):
    def __init__(self, org) -> None:
        super().__init__()
        self.org = org #obraz oryginalny

    def exe(self, weights, dir="charts/sharpened/"):
        lap_img = cv2.Laplacian(self.org, cv2.CV_64F) # obraz krawędziowy
        for weight in weights:
            img = self.org - weight * lap_img # dodanie ważonego obrazu kr.
            cv2.imwrite(f"{dir}sharp_w={weight}.png", img) # zapis




if __name__ == "__main__":
    t = tools()
    orginal = t.readimg("latarnia2_col.png")
    gnoised = t.readimg("latarnia2_col_noise.png")
    snoised = t.readimg("latarnia2_col_inoise.png")

    zad = zad1(orginal, snoised, gnoised)
    zad.executeGauss([3, 5, 7])
    zad.executeMedian([3, 5, 7])
    print(pd.DataFrame(np.array(zad.psnr)[1:], columns=np.array(zad.psnr)[0]))

    # Matplotlib malfunction with cv2 -> handling hist in other module
    zad = zad2(orginal)
    ans = zad.exe()
    with open("dump1.txt", "+w") as handle:
        for i in ans[0]:
            handle.writelines(str(i) + '\n')
    with open("dump2.txt", "+w") as handle:
        for i in ans[1]:
            handle.writelines(str(i) + '\n')


    zad = zad3(cv2.GaussianBlur(orginal, (3, 3), 0))
    zad.exe([0.1, 0.5, 1, 3])
