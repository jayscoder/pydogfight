from PIL import Image
import os

def convert_png_to_jpg(input_path, output_path):
    try:
        # 打开PNG图片
        with Image.open(input_path) as img:
            # 转换为RGB模式（JPG格式不支持透明度，所以需要转换为RGB）
            # 创建白色背景图像
            white_bg_img = Image.new('RGBA', img.size, (255, 255, 255))
            # 将PNG图片粘贴到白色背景图像上
            white_bg_img.paste(img, mask=img.split()[3])
            # 保存为JPG格式
            white_bg_img.convert('RGB').save(output_path)
        print(f'图片转换成功：{output_path}')
    except Exception as e:
        print(f'转换失败：{e}')


# 输入PNG图片路径和输出JPG图片路径
input_png_path = 'missile.png'
output_jpg_path = 'missile.jpg'

if __name__ == '__main__':
    convert_png_to_jpg(input_png_path, output_jpg_path)
