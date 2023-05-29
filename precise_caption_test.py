import vit
import translate_en2ja
import clip
import matplotlib.pyplot as plt
import japanize_matplotlib
import torch
import torch.nn.functional as F

if __name__ == "__main__":

    print("loading...")

    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Hiragino Maru Gothic Pro', 'Yu Gothic', 'Meirio', 'Takao', 'IPAexGothic', 'IPAPGothic', 'Noto Sans CJK JP']

    image_path = "./smr.jpeg"

    # en_cap = vit.predict([image_path])
    # ja_cap = translate_en2ja.translate([en_cap])

    # captions = [
    #     "和",         # 日本語 (Japanese)
    #     "harmony",    # 英語 (English)
    #     "日本",
    #     "Japanese",
    #     "Kyoto",
    #     "京都",
    #     "傘",
    #     "日傘",
    #     "番傘",
    #     "Umbrella",
    #     "Parasol",
    #     "英語リスニング",
    # ]

    # captions = [
    #     "寿司", "sushi", "sushis", "суши", "壽司", "스시", "和",
    #     "harmony", 
    #     "日本",
    #     "Japanese",
    #     "Kyoto",
    #     "京都",
    # ]

    # captions = [
    #     "ソンブレロ",
    #     "sombrero",
    #     "chapeau de paille",
    #     "сомбреро",
    #     "草帽",
    #     "솜브레로"
    # ]

    # captions = [
    #     "日本",
    #     "Japanese",
    #     "Kyoto",
    #     "京都",
    #     "忍び",
    #     "刺客",
    #     "닌자",
    #     "نينجا",
    #     "忍者",
    #     "נִינְג'ָה",
    #     "निंजा",
    #     "Ninja",
    # ]

    captions = [
        "アニメ",
        "Comic",
        "アニメキュラクター",
        "Anime character",
        "萌え",
        "Moe",
        "Anime song",
        "Original work",
        "Anime fan",
        "Anime broadcast",
        "Anime industry",
        "Anime studio",
    ]

    # print(captions)
    
    text_emb, img_emb = clip.get_embedding(image_path, captions)
    sim = F.cosine_similarity(img_emb, text_emb)
    print("cos similarity: ")
    print(sim)

    norms = []
    for t in text_emb:
        norms.append(torch.norm(t).item())
    norms.append(torch.norm(img_emb).item())
    
    #グラフを表示する領域を，figオブジェクトとして作成。
    fig = plt.figure(figsize = (10,6), facecolor='lightblue')

    #グラフを描画するsubplot領域を作成。
    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2)

    ax1.bar(captions + ["image"], norms)
    ax2.bar(captions, sim.detach().numpy())

    plt.show()