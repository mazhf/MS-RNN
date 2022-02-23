import cv2
import os


def avi2png(rd_pth, sv_pth):
    act_lis = os.listdir(rd_pth)
    for act in act_lis:
        video_name_lis = os.listdir(os.path.join(rd_pth, act))
        for video_name in video_name_lis:
            video_read_pth = os.path.join(rd_pth, act, video_name)
            vc = cv2.VideoCapture(video_read_pth)
            video_save_pth = os.path.join(sv_pth, act, video_name.split('.avi')[0][::-1].split('_', 1)[1][::-1])
            if not os.path.exists(video_save_pth):
                os.makedirs(video_save_pth)
            count = 1
            rat = True
            if vc.isOpened():
                print('视频读取成功，正在逐帧截取...')
                while rat:
                    rat, frame = vc.read()
                    if rat:
                        frame = cv2.resize(frame, (100, 100))
                        cv2.imwrite(os.path.join(video_save_pth, str(count) + '.png'), frame)
                        count += 1
                vc.release()
                print('截取完成，图像保存在：%s' % video_save_pth)
                print('************************************************')
            else:
                print('视频读取失败，请检查文件地址')


if __name__ == '__main__':

    read_pth = "/home/mazhf/Spatiotemporal/dataset/kth"
    save_pth = "/home/mazhf/Spatiotemporal/dataset/kth_resize_png"
    if not os.path.exists(save_pth):
        os.makedirs(save_pth)
    avi2png(read_pth, save_pth)
