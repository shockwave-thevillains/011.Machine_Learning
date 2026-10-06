# Naive Bayes: filter SMS spam berbahasa Indonesia
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

sms = [
    ("Selamat! Anda menang undian mobil, hubungi nomor ini sekarang", "spam"),
    ("PROMO pulsa gratis 100rb klik link berikut", "spam"),
    ("Pinjaman dana cepat tanpa jaminan cair hari ini", "spam"),
    ("Anda terpilih dapat hadiah uang tunai, transfer biaya admin", "spam"),
    ("Klik link untuk klaim hadiah saldo gratis", "spam"),
    ("Dapatkan bonus deposit 200% hanya hari ini", "spam"),
    ("Menang hadiah iphone, kirim data diri anda", "spam"),
    ("Kredit tanpa survey bunga rendah hubungi kami", "spam"),
    ("Nanti sore jadi rapat jam 4 di kantor?", "ham"),
    ("Bu, saya sudah sampai rumah", "ham"),
    ("Jangan lupa beli telur dan beras ya", "ham"),
    ("Tugas kelompok dikumpulkan besok pagi", "ham"),
    ("Makan siang bareng yuk di warung biasa", "ham"),
    ("Paket kamu sudah saya terima tadi siang", "ham"),
    ("Besok jemput adik di sekolah jam 12", "ham"),
    ("Terima kasih sudah datang ke acara kemarin", "ham"),
]
teks, label = zip(*sms)
model = make_pipeline(CountVectorizer(), MultinomialNB(alpha=1.0)).fit(teks, label)

uji = ["Gratis hadiah pulsa, klik link sekarang",
       "Rapat besok pagi di kantor ya",
       "Hubungi kami untuk pinjaman dana",
       "Sudah sampai sekolah, nanti jemput jam 12"]
for t, p, pr in zip(uji, model.predict(uji), model.predict_proba(uji)):
    print(f"[{p.upper():>4}] P(spam)={pr[1]:.3f} | {t}")

nb, vocab = model[-1], model[0].get_feature_names_out()
skor = nb.feature_log_prob_[1] - nb.feature_log_prob_[0]
print("Kata paling 'spam' :", list(vocab[skor.argsort()[-6:][::-1]]))
print("Kata paling 'ham'  :", list(vocab[skor.argsort()[:6]]))
