using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using SixLabors.ImageSharp.Processing.Processors.Transforms;

namespace Yolov7net.Extentions
{
    public static class Utils
    {
        /// <summary>
        /// xywh to xyxy
        /// </summary>
        /// <param name="source"></param>
        /// <returns></returns>
        public static float[] Xywh2xyxy(float[] source)
        {
            var result = new float[4];

            result[0] = source[0] - source[2] / 2f;
            result[1] = source[1] - source[3] / 2f;
            result[2] = source[0] + source[2] / 2f;
            result[3] = source[1] + source[3] / 2f;

            return result;
        }

        public static Image ResizeImage(Image image, int target_width, int target_height)
        {
            var (w, h) = (image.Width, image.Height); // image width and height
            var (xRatio, yRatio) = (target_width / (float)w, target_height / (float)h); // x, y ratios
            var ratio = Math.Min(xRatio, yRatio); // ratio = resized / original
            var (width, height) = ((int)(w * ratio), (int)(h * ratio)); // roi width and height
            var (x, y) = ((target_width / 2) - (width / 2), (target_height / 2) - (height / 2)); // roi x and y coordinates
            var roi = new Rectangle(x, y, width, height); // region of interest

            var options = new ResizeOptions
            {
                Size = new Size(target_width, target_height),
                TargetRectangle = roi,
                Sampler = KnownResamplers.Bicubic
            };
            return image.Clone(i => i.Resize(options));
        }

        public static Tensor<float> ExtractPixels(Image input)
        {
            using var image = input.CloneAs<Rgb24>();
            var rectangle = new Rectangle(0, 0, image.Width, image.Height);
            int bytesPerPixel = image.PixelType.BitsPerPixel / 8;

            var tensor = new DenseTensor<float>(new[] { 1, 3, image.Height, image.Width });

            image.ProcessPixelRows(accessor =>
            {
                for (int y = 0; y < accessor.Height; y++)
                {
                    Span<Rgb24> pixelSpan = accessor.GetRowSpan(y);
                    for (int x = 0; x < accessor.Width; x++)
                    {
                        tensor[0, 0, y, x] = pixelSpan[x].R / 255f;
                        tensor[0, 1, y, x] = pixelSpan[x].G / 255f;
                        tensor[0, 2, y, x] = pixelSpan[x].B / 255f;
                    }
                }
            });
            
            return tensor;
        }

        public static float Clamp(float value, float min, float max)
        {
            return value < min ? min : value > max ? max : value;
        }
    }
}
