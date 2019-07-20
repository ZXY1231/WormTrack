import os
import cv2 as cv
import numpy as np
import csv



class WorkTracker:
    worm_t = {} # {timecounter: y0, x0, img_wormbody, area, mean, std}
    window = 35
    worm_a_m_s = None
    worms_a = []
    worms_m = []
    worms_s = []
    spamwriter = None

    #def __init__(self, label, x1, y1, x2, y2, ori_imgs, out_dir):
    def __init__(self, label, x1, y1, x2, y2, out_dir):
        self.label = label
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2
        #self.ori_imgs = dict(ori_imgs)
        self.out_dir = out_dir
        self.out_body_dir = out_dir +'/body' # processed body image
        self.out_prev_dir = out_dir +'/prev' # only mask preview image

        if not os.path.isdir(self.out_body_dir):
            os.mkdir(self.out_body_dir)
        if not os.path.isdir(self.out_prev_dir):
            os.mkdir(self.out_prev_dir)
        header = ['worm_label', 'counter', 'y_0', 'x_0', 'img_name']
        self.csvfile = open(out_dir+ '/trajectory-%d.csv'%label, "w+",
                buffering=1)
        self.spamwriter = csv.writer(self.csvfile, delimiter='\t', quoting =
                csv.QUOTE_MINIMAL)
        self.spamwriter.writerow(header)
        print("create track worm", label)
        print("[", x1, y1, x2, y2, "]")

    def wormbody_pre(self, counter):
        area_p = -1
        mean_p = -1
        std_p = -1
        if 3< counter <= 15:
            worms_img = []
            worms_area = []
            for i in range(1, counter):
                worms_img.append(self.worm_t[i][2])
                worms_area.append(cv.countNonZero(self.worm_t[i][2]))
            while(len(worms_img) > 2 ): # At least 3 elements
                if max(worms_area) - min(worms_area) < 0.2*min(worms_area):
                    area_p = np.mean(worms_area)
                    mean_p_list = []
                    std_p_list = []
                    for img in worms_img:
                        mean_t, std_t = cv.meanStdDev(img, None, None,
                                (img>0).astype(np.uint8))
                        mean_p_list.append(mean_t)
                        std_p_list.append(std_t)
                    mean_p = np.mean(mean_p_list)
                    std_p  = np.mean(std_p_list)
                    break
                else:
                    if len(worms_img) > 0: # Be care here
                        min_area = min(worms_area)
                        max_area = max(worms_area)
                        if ((max_area) - np.mean(worms_area) <
                                (np.mean(worms_area) - min_area)):
                            index = worms_area.index(min_area)
                            worms_area.remove(min_area)
                            worms_img.pop(index)
                        else: # Remove max or min
                            index = worms_area.index(max_area)
                            worms_area.remove(max_area)
                            worms_img.pop(index)
        elif 100 > counter > 15: # counter_l > 15: # Process after frame 15
            window_size = 15
            search_shift = 15
            left = counter  - 15
            right = counter -1
            worms_img = []
            worms_area = []
            min_left = left - 1 - window_size - search_shift
            if min_left < 1:
                min_left = 1
            while(left > min_left):
                worms_area.clear()
                worms_img.clear()
                for i in range(left, right+1):
                    worms_img.append(self.worm_t[i][2])
                    worms_area.append(cv.countNonZero(self.worm_t[i][2]))
                if max(worms_area) - min(worms_area)< 0.2*min(worms_area):
                    area_p = np.mean(worms_area)
                    mean_p_list = []
                    std_p_list = []
                    for img in worms_img:
                        mean_t, std_t = cv.meanStdDev(img, None, None,
                                (img>0).astype(np.uint8))
                        mean_p_list.append(mean_t)
                        std_p_list.append(std_t)
                    mean_p = np.mean(mean_p_list)
                    std_p  = np.mean(std_p_list)
                    break
                else:
                    left -= 1
                    right-= 1
        elif counter > 100:
            area_size = 100
            mean_size = 50
            std_size = 25
            area_p_list = []
            std_p_list = []
            if (self.worm_t[counter-1][3] - self.worm_a_m_s[0])< 0.2 \
                    * self.worm_a_m_s[0] :
                for i in range(area_size):
                    area_p_list.append(self.worm_t[counter-1-i][4])
                area_p = np.mean(area_p_list)
            if abs(self.worm_t[counter-1][4] - self.worm_a_m_s[1])< 0.01 \
                    * self.worm_a_m_s[1]:
                mean_p_list = []
                for i in range(mean_size):
                    mean_p_list.append(self.worm_t[counter-1-i][4])
                mean_p = np.mean(mean_p_list)
            if (self.worm_t[counter-1][5] - self.worm_a_m_s[2])> -0.02 \
                    * self.worm_a_m_s[2]:
                for i in range(std_size):
                    std_p_list.append(self.worm_t[counter-1-i][5])
                std_p = np.mean(std_p_list)
        return area_p, mean_p, std_p


    def evalBody(self, img_wormbody_l, counter, img_whole, y_ori, x_ori):
        img_wormbody_mask = np.ones(img_wormbody_l.shape, dtype=np.uint16)
        img_wormbody_l_bi = cv.threshold(img_wormbody_l, 0, 1,
                cv.THRESH_BINARY)[1]
        connected = cv.connectedComponents(img_wormbody_l_bi.astype(np.uint8))

        previous = self.wormbody_pre(counter)
        if not( -1 in previous): # Update value only when it have valid value
            self.worm_a_m_s = previous
            print(self.worm_a_m_s)
        area_pre = self.worm_a_m_s[0]
        mean_pre = self.worm_a_m_s[1]
        std_pre = self.worm_a_m_s[2]
        self.worms_a.append(area_pre)
        self.worms_m.append(mean_pre)
        self.worms_s.append(std_pre)

        if connected[0] > 2: # only has one obejct
            l = self.window//2 # Diffusion
            x_ori_extend = x_ori - l
            y_ori_extend = y_ori - l
            img_roi_extend = img_whole[y_ori_extend : y_ori_extend + 4*l, \
                x_ori_extend : x_ori_extend + 4*l]
            img_wormbody_extend=np.zeros(img_roi_extend.shape,dtype=np.uint16)
            img_wormbody_extend[l: 3*l+1, l:3*l+1] = img_wormbody_l
            img_wormbody_extend_new = self.extendBody(img_roi_extend,
                    img_wormbody_extend, counter)
            img_wormbody_extend_new_bi = cv.threshold(img_wormbody_extend_new,
                    0, 1, cv.THRESH_BINARY)[1]
            connected = cv.connectedComponents(
                    img_wormbody_extend_new_bi.astype(np.uint8))

            flag1 = 0
            flag2 = 0
            min_mean_diff = 2**16
            min_std_diff = 2**16
            for conp in range(1, connected[0]):
                conp_mask = (connected[1] == conp).astype(np.uint8)
                area_conp = cv.countNonZero(conp_mask)
                if (area_conp - area_pre ) < 0.4 * area_pre:
                    mean, std = cv.meanStdDev(img_wormbody_extend_new, None,
                            None, conp_mask)
                    if abs(mean - mean_pre) < min_mean_diff :
                        flag1 = conp
                        min_mean_diff = abs(mean - mean_pre)
                    if abs(std - std_pre) < min_std_diff :
                        flag2 = conp
                        min_std_diff = abs(std - std_pre)
            if not (flag1 == 0):
                if flag1 == flag2:
                    img_wormbody_mask = (connected[1] == flag1)[l:3*l+1,
                        l:3*l+1].astype(np.uint16)
                else:
                    if (min_mean_diff / mean_pre) > (min_std_diff / std_pre):
                        img_wormbody_mask = (connected[1] == flag1)[l:3*l+1,
                            l:3*l+1].astype(np.uint16)
                    else:
                        img_wormbody_mask = (connected[1] == flag2)[l:3*l+1,
                            l:3*l+1].astype(np.uint16)
        return img_wormbody_mask

    # Initialize the seed from bounding boxa
    def tracker_init(self, img0):
        #img0 = self.ori_imgs.get(0)[0] # Read as origal format
        img0 = cv.imread(img0, -1)
        img0_roi = img0[self.y1:self.y2+1, self.x1:self.x2+1]
        img0_roi_th = (cv.GaussianBlur(img0_roi, (31,31), 0)
                        - cv.GaussianBlur(img0_roi, (3,3), 0))
        y_ori = self.y1
        x_ori = self.x1
        pos_wormbody = np.where( img0_roi_th > 10000)
        x_mid = int((max(pos_wormbody[1]) + min(pos_wormbody[1]))/2 + x_ori)
        y_mid = int((max(pos_wormbody[0]) + min(pos_wormbody[0]))/2 + y_ori)
        pos_wormbody = list(zip(pos_wormbody[0] + y_ori, pos_wormbody[1]+x_ori))
        l = self.window//2

        img_wormbody = img0[y_mid-l:y_mid+l+1, x_mid-l:x_mid+l+1]
        body_mask = np.zeros(img_wormbody.shape, dtype=np.uint16)
        for i in pos_wormbody:
            body_mask[i[0]-y_mid+l, i[1]-x_mid+l] = 1
        img_wormbody = img_wormbody * body_mask
        mean,std=cv.meanStdDev(img_wormbody,None,None,body_mask.astype(np.uint8))
        area = cv.countNonZero(img_wormbody)
        self.worm_a_m_s = (area, mean, std)
        self.worms_a.append(area)
        self.worms_m.append(mean)
        self.worms_s.append(std)
        self.worm_t[-1] = (y_mid-l, x_mid-l, img_wormbody, area, mean, std)
        #rows = [self.label, 0, y_mid-l, x_mid-l, self.ori_imgs[0][1]]

    # select points which intensity between 1/2~3/4 order
    def intensityMask(self, img_wormbody_l):
        intensity_mask = np.ones(img_wormbody_l.shape, dtype=np.uint16)
        img_wormbody_l_ravel = (np.copy(img_wormbody_l)).ravel()
        img_wormbody_l_ravel.sort()
        pos_nonzero = img_wormbody_l_ravel.nonzero()[0]
        pos_lowboundary = 0
        pos_upboundary  = len(pos_nonzero) - 1
        if(len(pos_nonzero) < 20):
            print("Too less wormbody pixel")
        else:
            pos_lowboundary = len(pos_nonzero) - 1 - len(pos_nonzero)//5
            pos_upboundary = len(pos_nonzero) - 1 - 3 # take out highest 3
        intensity_lowboundary=img_wormbody_l_ravel[pos_nonzero[pos_lowboundary]]
        intensity_upboundary=img_wormbody_l_ravel[ pos_nonzero[pos_upboundary]]
        # Error handle when boundary is not acceptable: too low,high, no value
        intensity_mask=cv.threshold(img_wormbody_l, intensity_upboundary, 65535,
            cv.THRESH_TOZERO_INV)[1]
        intensity_mask = cv.threshold(intensity_mask, intensity_lowboundary, 1,
            cv.THRESH_BINARY)[1]
        # Error handler: no active point
        return intensity_mask

    # Extend wormbody using previous wormbody position
    # Extend wormbody by finding connected component: X_k = ( X_{k-1} ⊕ SE )^N
    def extendBody(self, img_new_roi_l, img_wormbody_old_l, counter_l, I=False):
        # Difference of Gaussian: erode a ring between wormbody

        N = (cv.GaussianBlur(img_new_roi_l, (31,31), 0)
                -cv.GaussianBlur(img_new_roi_l, (3,3), 0))
        N = cv.threshold(N, 10000, 1, cv.THRESH_BINARY)[1]
        img_wormbody_old_l_bi = cv.threshold(img_wormbody_old_l, 0, 1,
        cv.THRESH_BINARY)[1]
        # Only search wormbody in this region to avoid noise
        if I :
            # Limit search window's boundary away from original edge 3 pixels
            N_mask_p1 = cv.dilate(img_wormbody_old_l_bi, np.ones([7,7]))
            N = N * N_mask_p1
        element = np.ones([3,3])
        # Initial points to start extend wormbody
        X1_mask_intensity = self.intensityMask(img_wormbody_old_l)
        seed = cv.erode(img_wormbody_old_l_bi, element) * X1_mask_intensity * N
        X1 = seed
        X0 = np.zeros(X1.shape, dtype=np.uint16)
        while(sum(sum( X0 - X1))):
            X0 = X1
            X1 = (cv.dilate(X0, element)) * N
        img_wormbody_new_l=X1*img_new_roi_l #* evalBody(X1, worm_l, counter_l)
        # TODO: when mask is empty
        return img_wormbody_new_l

    ## Return edge mask: edge_mask = binary(img_src) - [binary(img_src) ⊖ SE]
    def edgeScanner(self, img_src):
        ret, img_binary = cv.threshold(img_src, 0, 1, cv.THRESH_BINARY)
        element = cv.getStructuringElement(cv.MORPH_RECT,(3,3))
        edge_mask = img_binary - cv.erode(img_binary, element)
        return edge_mask

    # Main tracking process
    def tracking(self, ori_imgs):
        print("tracking:", self.label)
        #total = len(self.ori_imgs)
        total = len(ori_imgs)
        l = self.window//2
        for counter in range(total):
            y_ori, x_ori = self.worm_t.get(counter-1)[0:2]
            img_wormbody_old = self.worm_t.get(counter-1)[2]
            #img_new = self.ori_imgs[counter][0]
            #img_new = ori_imgs[counter][0]
            #img_name = self.ori_imgs[counter][1]
            #img_name = ori_imgs[counter][1]
            img_name = ori_imgs[counter][1]
            img_new = cv.imread(ori_imgs[counter][0],-1)
            img_new_roi = img_new[y_ori : y_ori+self.window,
                    x_ori : x_ori+self.window]
            img_wormbody = self.extendBody(img_new_roi,img_wormbody_old,counter)
            
            # Recenter wormbody
            pos_wormbody = np.where( img_wormbody > 0)
            if (len(pos_wormbody[0] > 1)): # Avoid unacceptable result
                y_mid = (min(pos_wormbody[0]) + max(pos_wormbody[0]))//2 +y_ori
                x_mid = (min(pos_wormbody[1]) + max(pos_wormbody[1]))//2 +x_ori
                y_ori_new = y_mid -l
                x_ori_new = x_mid -l
                mask_edge = self.edgeScanner(img_wormbody)
                img_wormbody_mask = np.zeros(img_wormbody.shape,dtype=np.uint16)
                pos_wormbody = list(zip(pos_wormbody[0]+y_ori, pos_wormbody[1]
                    + x_ori))
                for i in pos_wormbody:
                    img_wormbody_mask[i[0]-y_ori_new, i[1]-x_ori_new] = 1
                img_wormbody = img_wormbody_mask * img_new[ \
                        y_ori_new : y_mid+l+1, x_ori_new : x_mid+l+1]
                img_wormbody = img_wormbody * self.evalBody(img_wormbody,
                        counter, img_new, y_ori_new, x_ori_new)
                # TODO: error handle
                img_prev = img_new[y_mid-50 : y_mid+50, 
                        x_mid-50 : x_mid+50].copy()
                #img_prev = ori_imgs[counter][0][].copy()
                p1 = (y_ori-y_mid+50, x_ori+50-x_mid)
                p2 = (p1[0]+self.window, p1[1]+self.window)
                img_prev[p1[0]:p2[0], p1[1]:p2[1]] = img_prev[p1[0]:p2[0],
                        p1[1]:p2[1]]* (1-mask_edge)
                cv.rectangle(img_prev, p1, p2, (65536, 65536, 65536), 1, 
                        cv.LINE_4)
                cv.imwrite(self.out_body_dir + '/' + img_name, img_wormbody)
                cv.imwrite(self.out_prev_dir + '/' + img_name, img_prev)
                
                mean_std = cv.meanStdDev(img_wormbody, None, None,
                    (img_wormbody>0).astype(np.uint8))
                self.worm_t[counter] = (y_ori_new, x_ori_new, img_wormbody,
                    cv.countNonZero(img_wormbody), mean_std[0], mean_std[1])
                row = [self.label, counter, y_ori_new, x_ori_new, img_name]
                self.spamwriter.writerow(row)

            else:
                print("Track fail")
                break # Just break from now to avoid block process
            print(self.label, "-", counter)
        self.csvfile.close()
        return self.worm_t




