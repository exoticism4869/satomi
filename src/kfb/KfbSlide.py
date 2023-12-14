from PIL import Image
import satomi.kfb.kfbReader as kr
import cv2


class KfbSlide:
    """实现与openslide相同的常用接口,支持对kfb格式的切片图像进行读取

    kr.reader()返回的对象有以下几个方法:
        self.slide.ReadInfo(path, scale=0, ReadAll=False)
            path: Kfb文件路径\n
            scale: 读取倍数,0和40表示原始大小\n
            ReadAll: 暂时没有发现设为True和False在程序运行中有什么区别
        self.slide.getReadScale()
            始终返回40
        self.slide.getWidth(), self.slide.getHeight()
            返回的是ReadInfo设置的scale下的宽高
        self.slide.ReadRoi(x, y, width, height, scale)
            x, y: 对应scale下的坐标\n
            width, height: 对应scale下的宽高\n
            scale: 读取倍数,0和40表示原始大小
        self.slide.setReadScale(scale)
            scale: 设置合理的scale可以大大加快读取速度,但设置较高层也可以对低层进行读取
    由于kr.reader()返回的对象没有与level相关的api,故该class只实现了get_thumbnail和read_region两个最常用的方法

    Attributes:
    ----------------------------------------------------------------
    slide (reader): 可以调用kr.reader()返回的对象的方法.\n
    dimensions (tuple): A (width, height) tuple for level 0 of the slide.
    """

    def __init__(self, path):
        self.slide = kr.reader()
        self.slide.ReadInfo(path, 0)

        self.dimensions = (self.slide.getWidth(), self.slide.getHeight())
        self.level_dimensions = []

    def get_thumbnail(self, size: tuple):
        """Return an Image containing an RGB thumbnail of the slide.

        若缩小倍数小于等于8,则读取level3的thumb,再resize到对应倍数下的thumb(内存限制)\n
        若缩小倍数大于8,直接读取对应level的thumb

        Parameters:
        ----------------------------------------------------------------
        size (tuple): the maximum size of the thumbnail as a (width, height) tuple

        Returns:
        ----------------------------------------------------------------
        PIL.Image: an RGB image containing the contents of the slide
        """
        downsample = size[0] // self.dimensions[0]
        if downsample <= 8:
            self.slide.setReadScale(40 / 8)
            thumb = self.slide.ReadRoi(
                0, 0, self.dimensions[0] // 8, self.dimensions[1] // 8, 40 / 8
            )
            thumb = cv2.resize(thumb, (size[0], size[1]))
        else:
            self.slide.setReadScale(40 / downsample)
            thumb = self.slide.ReadRoi(
                0,
                0,
                self.dimensions[0] // downsample,
                self.dimensions[1] // downsample,
                40 / downsample,
            )
        thumb = cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB)
        return Image.fromarray(thumb)

    def read_region(self, location, level, size):
        """Return an RGBA Image containing the contents of the specified region.

        Parameters:
        ----------------------------------------------------------------
            location (tuple): The top left pixel (x,y) in the level 0 reference frame.
            level (int): The level number.
            size (tuple): The region size (width, height) on the given level.

        Returns:
        ----------------------------------------------------------------
        PIL.Image: an RGBA image containing the contents of the region.
        """

        scale = 40 / (2**level)
        roi_x, roi_y = location[0] // (2**level), location[1] // (2**level)
        roi_width, roi_height = size[0], size[1]
        patch = self.slide.ReadRoi(roi_x, roi_y, roi_width, roi_height, scale)
        patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGBA)
        return Image.fromarray(patch)


if __name__ == "__main__":
    kfb_path = "/data/lymphnode/tiff/泛癌淋巴结/4肺癌-淋巴结/0112318-OK/0112318D.kfb"
    kfb_slide = KfbSlide(kfb_path)
    thumb = kfb_slide.get_thumbnail(
        (kfb_slide.dimensions[0] // 30, kfb_slide.dimensions[1] // 30)
    )
    thumb.save("/home/hdc/Sebs/satomi/thumb.jpg")

    # patch = kfb_slide.read_region((10000, 20000), 1, (1024, 1024))
    # patch.save("/home/hdc/Sebs/satomi/patch.png")
