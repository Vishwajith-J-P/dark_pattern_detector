from PIL import Image, ImageDraw

for size in [16, 48, 128]:
    img  = Image.new('RGBA', (size, size), (15, 23, 42, 255))
    draw = ImageDraw.Draw(img)
    margin = size // 6
    draw.ellipse([margin, margin, size-margin, size-margin],
                 outline=(56, 189, 248, 255), width=max(1, size//12))
    img.save(f'icons/icon{size}.png')
    print(f'Created icon{size}.png')

