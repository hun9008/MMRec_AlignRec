from PIL import Image, ImageDraw, ImageFont

# 병합할 이미지 경로
# left_path = 'tsne_ALIGNREC_3view_top10.png'
# right_path = 'tsne_ALIGNREC_ANCHOR_1101_3view_top10.png'
# output_path = 'tsne_compare_concat.png'

# left_path = 'visualize_alignrec_vs_anchor_3529.png'
# right_path = 'visualize_alignrec_vs_anchor_5016.png'
# output_path = 'figure4_concat.png'

# left_title = "a) Neighbor Comparison – Text Case"
# right_title = "b) Neighbor Comparison – Image Case"

left_path = 'overlap_k20_alignrec.png'
right_path = 'overlap_k20_anchor.png'
output_path = 'figure5_concat.png'

left_title = "a) i-i Similarity Overlap for AlignRec"
right_title = "b) i-i Similarity Overlap for AnchorRec"

# left_path = 'rec_overlap_k20.png'
# right_path = 'rec_overlap_anchor_k20.png'
# output_path = 'figure6_concat.png'

# left_title = "a) Recommendation Overlap for AlignRec"
# right_title = "b) Recommendation Overlap for AnchorRec"

left_img = Image.open(left_path)
right_img = Image.open(right_path)

# (옵션) 두 이미지 높이 맞추기
if left_img.height != right_img.height:
    common_h = min(left_img.height, right_img.height)
    left_img = left_img.resize(
        (int(left_img.width * common_h / left_img.height), common_h)
    )
    right_img = right_img.resize(
        (int(right_img.width * common_h / right_img.height), common_h)
    )

img_h = max(left_img.height, right_img.height)
total_w = left_img.width + right_img.width

# 폰트 로드 (아무거나, 없으면 기본 폰트)
try:
    font_size = int(img_h * 0.05)  # 이미지 높이의 5% 정도
    font = ImageFont.truetype("DejaVuSans.ttf", font_size - 10)
except:
    font = ImageFont.load_default()

# 텍스트 크기 측정용 draw
tmp_img = Image.new("RGB", (10, 10))
tmp_draw = ImageDraw.Draw(tmp_img)

left_bbox  = tmp_draw.textbbox((0, 0), left_title,  font=font)
right_bbox = tmp_draw.textbbox((0, 0), right_title, font=font)
left_w, left_h   = left_bbox[2] - left_bbox[0], left_bbox[3] - left_bbox[1]
right_w, right_h = right_bbox[2] - right_bbox[0], right_bbox[3] - right_bbox[1]

padding = 20
title_h = max(left_h, right_h) + 2 * padding
total_h = img_h + title_h

# 새 캔버스 생성
new_img = Image.new("RGB", (total_w, total_h), (255, 255, 255))
draw = ImageDraw.Draw(new_img)

# 이미지 붙이기 (위쪽)
new_img.paste(left_img, (0, 0))
new_img.paste(right_img, (left_img.width, 0))

# 타이틀 위치 (아래쪽 중앙)
title_y = img_h + padding

left_title_x  = (left_img.width - left_w) // 2
right_title_x = left_img.width + (right_img.width - right_w) // 2

draw.text((left_title_x,  title_y), left_title,  fill=(0, 0, 0), font=font)
draw.text((right_title_x, title_y), right_title, fill=(0, 0, 0), font=font)

new_img.save(output_path)
print(f"병합 완료: {output_path}")