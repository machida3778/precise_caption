import vit
import translate_en2ja
import clip
import matplotlib.pyplot as plt

if __name__ == "__main__":

    print("loading...")

    image_path = "./wa3.jpeg"

    # en_cap = vit.predict([image_path])
    # ja_cap = translate_en2ja.translate([en_cap])

    captions = [
        "和",         # 日本語 (Japanese)
        "harmony",    # 英語 (English)
        "harmonie",   # フランス語 (French)
        "Harmonie",   # ドイツ語 (German)
        "armonía",    # スペイン語 (Spanish)
        "гармония",   # ロシア語 (Russian)
        "조화"         # 韓国語 (Korean)
    ]
    print(captions)
    
    sim = clip.calculate_similarity(image_path, captions)
    print(sim)
    print(captions[sim.argmax()])

    plt.bar(captions, sim.detach().numpy()[0])
    plt.show()