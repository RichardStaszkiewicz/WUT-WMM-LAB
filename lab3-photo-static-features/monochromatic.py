import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

class every_image(object):
    def __init__(self, dir: str) -> None:
        self.dir = dir
        self.image = cv2.imread(dir, cv2.IMREAD_UNCHANGED)
        print(f"image, wymiary: {self.image.shape}, typ danych:" +
            f"{self.image.dtype}, wartości: {self.image.min()} - {self.image.max()}")

    def bit_per_px(self):
        return 8 * os.path.getsize(self.dir) / (self.image.shape[0] * self.image.shape[1])

    def calc_entropy(self, hist):
        pdf = hist/hist.sum()
        entropy = -sum([x*np.log2(x) for x in pdf if x != 0])
        return entropy

    def printi(self, img=None, img_title="image"):
        img = self.image if img is None else img
        print(f"{img_title}, wymiary: {img.shape}, typ danych: {img.dtype}, wartości: {img.min()} - {img.max()}")

    def cv_imshow(self, img=None, img_title="image", dir=None):
        img = self.image if img is None else img
        dir = "./imshow.png" if dir is None else dir
        # if (img.dtype == np.float32) or (img.dtype == np.float64):
        #     img_ = img / 255
        # # elif img.dtype == np.int16:
        # #     img_ = img*128
        # else:
        img_ = img
        cv2.imwrite(dir, img_) # zapis

    def calc_hist(self, img=None):
        img = self.image if img is None else img
        hist_image = cv2.calcHist([self.image], [0], None, [256], [0, 256])
        hist_image = hist_image.flatten()
        return hist_image

    def error_hist_save(self, hist, dir):
        with open(dir, "+w") as handle:
            for i in hist:
                handle.writelines(str(i) + '\n')

    def dwt(self, img):
        """
        Bardzo prosta i podstawowa implementacja, nie uwzględniająca efektywnych metod obliczania DWT
        i dopuszczająca pewne niedokładności.
        """
        maskL = np.array([0.02674875741080976, -0.01686411844287795, -0.07822326652898785, 0.2668641184428723,
            0.6029490182363579, 0.2668641184428723, -0.07822326652898785, -0.01686411844287795, 0.02674875741080976])
        maskH = np.array([0.09127176311424948, -0.05754352622849957, -0.5912717631142470, 1.115087052456994,
            -0.5912717631142470, -0.05754352622849957, 0.09127176311424948])

        bandLL = cv2.sepFilter2D(img,         -1, maskL, maskL)[::2, ::2]
        bandLH = cv2.sepFilter2D(img, cv2.CV_16S, maskL, maskH)[::2, ::2] ### ze względu na filtrację górnoprzepustową -> wartości ujemne, dlatego wynik 16-bitowy ze znakiem
        bandHL = cv2.sepFilter2D(img, cv2.CV_16S, maskH, maskL)[::2, ::2]
        bandHH = cv2.sepFilter2D(img, cv2.CV_16S, maskH, maskH)[::2, ::2]

        return bandLL, bandLH, bandHL, bandHH



class monochromatic(every_image):
    def __init__(self, dir) -> None:
        super().__init__(dir)

    def calc_entropy(self):
        H_image = super().calc_entropy(self.calc_hist())
        return H_image

    def differential_image(self):
        img_tmp1 = self.image[:, 1:]
        img_tmp2 = self.image[:, :-1]
        image_hdiff = cv2.addWeighted(img_tmp1, 1, img_tmp2, -1, 0, dtype=cv2.CV_16S)
        image_hdiff_0 = cv2.addWeighted(self.image[:, 0], 1, 0, 0, -127, dtype=cv2.CV_16S)
        image_hdiff = np.hstack((image_hdiff_0, image_hdiff))
        self.printi(image_hdiff)
        self.cv_imshow(image_hdiff + 128, "image_hdiff", dir="lab3-photo-static-features/charts/differential/mono_diff.png")

        hist_image = self.calc_hist()
        hist_diff = cv2.calcHist([(image_hdiff+255).astype(np.uint16)], [0], None, [511], [0, 511]).flatten()

        self.error_hist_save(hist_image, "lab3-photo-static-features/mono_org_hist.txt")
        self.error_hist_save(hist_diff, "lab3-photo-static-features/mono_diff-hist.txt")

        return super().calc_entropy(hist_diff)

    def dwt_description(self):
        ll, lh, hl, hh = self.dwt(self.image)

        self.cv_imshow(ll, "LL2", "lab3-photo-static-features/charts/wavelet/LL.png")
        self.cv_imshow(cv2.multiply(lh, 2) + 128, "LH2", "lab3-photo-static-features/charts/wavelet/LH.png")
        self.cv_imshow(cv2.multiply(hl, 2) + 128, "HL2", "lab3-photo-static-features/charts/wavelet/HL.png")
        self.cv_imshow(cv2.multiply(hh, 2) + 128, "HH2", "lab3-photo-static-features/charts/wavelet/HH.png")

        hist_ll = cv2.calcHist([ll], [0], None, [256], [0, 256]).flatten()
        hist_lh = cv2.calcHist([(lh+255).astype(np.uint16)], [0], None, [511], [0, 511]).flatten()
        hist_hl = cv2.calcHist([(hl+255).astype(np.uint16)], [0], None, [511], [0, 511]).flatten()
        hist_hh = cv2.calcHist([(hh+255).astype(np.uint16)], [0], None, [511], [0, 511]).flatten()
        H_ll = super().calc_entropy(hist_ll)
        H_lh = super().calc_entropy(hist_lh)
        H_hl = super().calc_entropy(hist_hl)
        H_hh = super().calc_entropy(hist_hh)

        self.error_hist_save(hist_ll, "lab3-photo-static-features/hist_ll.txt")
        self.error_hist_save(hist_lh, "lab3-photo-static-features/hist_lh.txt")
        self.error_hist_save(hist_hl, "lab3-photo-static-features/hist_hl.txt")
        self.error_hist_save(hist_hh, "lab3-photo-static-features/hist_hh.txt")
        # self.printi(ll, "LL")
        # self.printi(lh, "LH")
        # self.printi(hl, "HL")
        # self.printi(hh, "HH")

        return [H_ll, H_lh, H_hl, H_hh]


class colourful(every_image):
    def __init__(self, dir: str) -> None:
        super().__init__(dir)

    def entropy(self):
        image_R = self.image[:, :, 2] ### cv2.imread() zwraca obrazy w formacie BGR
        image_G = self.image[:, :, 1]
        image_B = self.image[:, :, 0]

        hist_R = cv2.calcHist([image_R], [0], None, [256], [0, 256]).flatten()
        hist_G = cv2.calcHist([image_G], [0], None, [256], [0, 256]).flatten()
        hist_B = cv2.calcHist([image_B], [0], None, [256], [0, 256]).flatten()

        return [self.calc_entropy(hist_R), self.calc_entropy(hist_G), self.calc_entropy(hist_B)]

    def YUV(self):
        image_YUV = cv2.cvtColor(self.image, cv2.COLOR_BGR2YUV)
        self.printi(image_YUV, "image_YUV")

        hist_Y = cv2.calcHist([image_YUV[:, :, 0]], [0], None, [256], [0, 256]).flatten()
        hist_U = cv2.calcHist([image_YUV[:, :, 1]], [0], None, [256], [0, 256]).flatten()
        hist_V = cv2.calcHist([image_YUV[:, :, 2]], [0], None, [256], [0, 256]).flatten()

        H_Y = self.calc_entropy(hist_Y)
        H_U = self.calc_entropy(hist_U)
        H_V = self.calc_entropy(hist_V)

        self.error_hist_save(hist_Y, "lab3-photo-static-features/col_Y-hist.txt")
        self.error_hist_save(hist_U, "lab3-photo-static-features/col_U-hist.txt")
        self.error_hist_save(hist_V, "lab3-photo-static-features/col_V-hist.txt")
        return [H_Y, H_U, H_V]

    def RGB(self, img=None):
        img = self.image if img is None else img
        hist_B = cv2.calcHist([img[:, :, 0]], [0], None, [256], [0, 256]).flatten()
        hist_G = cv2.calcHist([img[:, :, 1]], [0], None, [256], [0, 256]).flatten()
        hist_R = cv2.calcHist([img[:, :, 2]], [0], None, [256], [0, 256]).flatten()

        H_B = self.calc_entropy(hist_B)
        H_G = self.calc_entropy(hist_G)
        H_R = self.calc_entropy(hist_R)

        self.error_hist_save(hist_B, "lab3-photo-static-features/col_B-hist.txt")
        self.error_hist_save(hist_G, "lab3-photo-static-features/col_G-hist.txt")
        self.error_hist_save(hist_R, "lab3-photo-static-features/col_R-hist.txt")
        return [H_B, H_G, H_R]

    def RD_curve(self, qualities):
        xx = [] ### tablica na wartości osi X -> bitrate
        ym = [] ### tablica na wartości osi Y dla MSE
        yp = [] ### tablica na wartości osi Y dla PSNR
        for quality in qualities: ### wartości dla parametru 'quality' należałoby dobrać tak, aby uzyskać 'gładkie' wykresy...
            out_file_name = f"lab3-photo-static-features/charts/color/out_image_q{quality:03d}.jpg"
            cv2.imwrite(out_file_name, self.image, (cv2.IMWRITE_JPEG_QUALITY, quality))
            image_compressed = cv2.imread(out_file_name, cv2.IMREAD_UNCHANGED)
            bitrate = 8*os.stat(out_file_name).st_size/(self.image.shape[0]*self.image.shape[1])
            mse, psnr = self.calc_mse_psnr(self.image, image_compressed)
            xx.append(bitrate)
            ym.append(mse)
            yp.append(psnr)

        self.error_hist_save(xx, "lab3-photo-static-features/xx.txt")
        self.error_hist_save(ym, "lab3-photo-static-features/ym.txt")
        self.error_hist_save(yp, "lab3-photo-static-features/yp.txt")


    def calc_mse_psnr(self, img1, img2):
        imax = 255.**2
        mse = ((img1.astype(np.float64)-img2)**2).sum()/img1.size ###img1.size - liczba elementów w img1, ==img1.shape[0]*img1.shape[1] dla obrazów mono, ==img1.shape[0]*img1.shape[1]*img1.shape[2] dla obrazów barwnych
        psnr = 10.0*np.log10(imax/mse)
        return (mse, psnr)


if __name__ == "__main__":
    mono = monochromatic("lab3-photo-static-features/latarnia2_mono.png")
    print(f"a) Bits per pixle in PNG compression: {mono.bit_per_px()} bit/px")
    print(f"b) H(image) = {mono.calc_entropy():.4f}")
    # NIe, przepływność mniejsza nie oznacza że entropia jest większa od średniej
    # długości, ponieważ matma się nie zgadza i wgl indukowałoby to bit niosący więcej informacji niż 1
    print(f"c) H(differential) = {mono.differential_image():.4f}")
    entropy = mono.dwt_description()
    print(f"H(LL) = {entropy[0]:.4f} \nH(LH) = {entropy[1]:.4f} \nH(HL) = {entropy[2]:.4f} \nH(HH) = {entropy[3]:.4f} \nH_śr = {sum(entropy)/len(entropy):.4f}")

    col = colourful("lab3-photo-static-features/latarnia2_col.png")
    entr = col.entropy()
    print(f"a) RGB entropy:\nH(R) = {entr[0]:.4f} \nH(G) = {entr[1]:.4f} \nH(B) = {entr[2]:.4f} \nH_śr = {(sum(entr))/len(entr):.4f}")
    entr = col.YUV()
    print(f"b) YUV entropy:\nH(Y) = {entr[0]:.4f} \nH(U) = {entr[1]:.4f} \nH(V) = {entr[2]:.4f} \nH_śr = {(sum(entr))/len(entr):.4f}")
    entr = col.RGB()
    print(f"c) RGB entropy:\nH(B) = {entr[0]:.4f} \nH(G) = {entr[1]:.4f} \nH(R) = {entr[2]:.4f} \nH_śr = {(sum(entr))/len(entr):.4f}")

    col.RD_curve([90, 80, 70, 60, 50, 40, 30, 20, 10])
