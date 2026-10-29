import streamlit as st
import sqlite3, hashlib, os, io, json
import numpy as np
from PIL import Image
from itertools import combinations

st.set_page_config(page_title="AI Wardrobe", page_icon="🎀", layout="wide")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(BASE_DIR, "wardrobe.db")
IMG_DIR = os.path.join(BASE_DIR, "wardrobe_images")
os.makedirs(IMG_DIR, exist_ok=True)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');
html, body, [class*="css"] {font-family:Poppins,sans-serif}
.stApp {background:#fff5fa}
[data-testid="stSidebar"] {background:#ffe0ed}
h1,h2,h3 {color:#ad3974}
.stButton>button,.stFormSubmitButton>button {background:#e65b9b;color:white;border:0;border-radius:12px;font-weight:600;min-height:45px}
.stButton>button:hover,.stFormSubmitButton>button:hover {background:#c83c80;color:white}
.hero {background:linear-gradient(120deg,#ffd4e7,#fff0f7);padding:24px;border-radius:22px;margin-bottom:20px}
</style>
""", unsafe_allow_html=True)

CATEGORY = ["Top", "Bottom", "Dress", "Footwear", "Accessory", "Other"]
TYPES = ["Kurti", "Shirt", "T-shirt", "Top", "Jeans", "Pant", "Leggings",
         "Skirt", "Palazzo", "Dress", "Saree", "Salwar", "Dupatta",
         "Jacket", "Shoes", "Sandals", "Bag", "Other"]
COLORS = ["Black", "White", "Pink", "Blue", "Red", "Green", "Yellow",
          "Purple", "Brown", "Beige", "Grey", "Orange", "Multicolour", "Other"]
STYLES = ["Casual", "Formal", "Traditional", "Party", "Any"]
OCCASIONS = ["College", "Office", "Wedding", "Party", "Casual outing", "Festival", "Any"]

def add_missing_columns(con, table, wanted):
    existing = {r["name"] for r in con.execute(f"PRAGMA table_info({table})").fetchall()}
    for col, typ in wanted.items():
        if col not in existing:
            con.execute(f"ALTER TABLE {table} ADD COLUMN {col} {typ}")

def db_connect():
    con = sqlite3.connect(DB, timeout=30)
    con.row_factory = sqlite3.Row
    con.execute("CREATE TABLE IF NOT EXISTS users (username TEXT PRIMARY KEY, password TEXT NOT NULL)")
    con.execute("""CREATE TABLE IF NOT EXISTS clothes (
        id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT, name TEXT, category TEXT,
        type TEXT, color TEXT, style TEXT, occasions TEXT, image_path TEXT, clip_vector TEXT)""")
    con.execute("""CREATE TABLE IF NOT EXISTS feedback (
        id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT, outfit_key TEXT, rating INTEGER)""")
    add_missing_columns(con, "users", {"username":"TEXT", "password":"TEXT"})
    add_missing_columns(con, "clothes", {"id":"INTEGER", "username":"TEXT", "name":"TEXT",
        "category":"TEXT", "type":"TEXT", "color":"TEXT", "style":"TEXT",
        "occasions":"TEXT", "image_path":"TEXT", "clip_vector":"TEXT"})
    add_missing_columns(con, "feedback", {"id":"INTEGER", "username":"TEXT",
        "outfit_key":"TEXT", "rating":"INTEGER"})
    con.commit()
    return con

try:
    con = db_connect()
except Exception as e:
    st.error("Database could not be opened.")
    st.code(f"{type(e).__name__}: {e}")
    st.stop()

def pw_hash(password):
    return hashlib.sha256(password.encode()).hexdigest()

def full_path(path):
    if not path:
        return ""
    return path if os.path.isabs(path) else os.path.join(BASE_DIR, path)

def show_image(path, width=None):
    try:
        st.image(path, width=width if width else "stretch")
    except TypeError:
        st.image(path, width=width, use_container_width=width is None)

@st.cache_resource(show_spinner="Loading CLIP AI model...")
def get_clip():
    from transformers import CLIPModel, CLIPProcessor
    import torch
    model_id = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(model_id)
    processor = CLIPProcessor.from_pretrained(model_id)
    model.eval()
    return model, processor, torch

def extract_clip(image_bytes):
    model, processor, torch = get_clip()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        features = model.get_image_features(**inputs)
    if not hasattr(features, "norm"):
        features = features.pooler_output
    features = features / features.norm(p=2, dim=-1, keepdim=True)
    return features[0].cpu().numpy().tolist()

def similarity(a, b):
    a, b = np.array(a), np.array(b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom else 0.0

@st.cache_data(show_spinner=False)
def extract_pdf_images(pdf_bytes):
    """Extract separate embedded images from a PDF; render a page if it has no images."""
    import fitz
    result, seen = [], set()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    for page_no, page in enumerate(doc):
        found = False
        for info in page.get_images(full=True):
            try:
                raw = doc.extract_image(info[0])["image"]
                image = Image.open(io.BytesIO(raw)).convert("RGB")
                out = io.BytesIO()
                image.save(out, format="JPEG")
                data = out.getvalue()
                digest = hashlib.sha256(data).hexdigest()
                if digest not in seen:
                    seen.add(digest)
                    result.append((f"PDF page {page_no + 1} image {len(result) + 1}.jpg", data))
                found = True
            except Exception:
                continue
        if not found:
            pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False)
            data = pix.tobytes("png")
            digest = hashlib.sha256(data).hexdigest()
            if digest not in seen:
                seen.add(digest)
                result.append((f"PDF page {page_no + 1}.png", data))
    doc.close()
    return result

def add_uploads_to_pending(image_files, pdf_files):
    pending = []
    for f in image_files or []:
        try:
            data = f.getvalue()
            Image.open(io.BytesIO(data)).verify()
            pending.append((f.name, data))
        except Exception:
            st.warning(f"Could not read image: {f.name}")
    for f in pdf_files or []:
        try:
            extracted = extract_pdf_images(f.getvalue())
            if not extracted:
                st.warning(f"No images found in {f.name}.")
            else:
                st.caption(f"{f.name}: extracted {len(extracted)} image(s).")
                pending.extend((f"{f.name} — {name}", data) for name, data in extracted)
        except Exception as e:
            st.error(f"Could not read PDF {f.name}: {e}")
    return pending

st.markdown("""
<div class="hero">
<h1>🎀 AI Wardrobe</h1>
<h3>Your personal virtual fashion assistant</h3>
<p>Discover outfit combinations from your own wardrobe.</p>
</div>
""", unsafe_allow_html=True)

if "user" not in st.session_state:
    st.session_state.user = None
if "page" not in st.session_state:
    st.session_state.page = "Home"

def go_to(page):
    st.session_state.page = page

if st.session_state.user is None:
    tab_login, tab_register = st.tabs(["Login", "Register"])
    with tab_register:
        st.subheader("Create your account 💗")
        with st.form("register_form"):
            new_user = st.text_input("Choose username")
            new_password = st.text_input("Choose password", type="password")
            confirm = st.text_input("Confirm password", type="password")
            register = st.form_submit_button("Register")
        if register:
            if len(new_user.strip()) < 3:
                st.warning("Username needs at least 3 characters.")
            elif len(new_password) < 6:
                st.warning("Passwords need at least 6 characters.")
            elif new_password != confirm:
                st.warning("Passwords do not match.")
            else:
                try:
                    con.execute("INSERT INTO users(username,password) VALUES(?,?)",
                                (new_user.strip(), pw_hash(new_password)))
                    con.commit()
                    st.success("Account created! Now select the Login tab.")
                except sqlite3.IntegrityError:
                    st.error("Username already exists. Choose another.")
                except sqlite3.Error as e:
                    st.error(f"Database error: {e}")
    with tab_login:
        st.subheader("Welcome back 💕")
        with st.form("login_form"):
            login_user = st.text_input("Username")
            login_password = st.text_input("Password", type="password")
            login = st.form_submit_button("Login")
        if login:
            try:
                result = con.execute("SELECT * FROM users WHERE username=? AND password=?",
                    (login_user.strip(), pw_hash(login_password))).fetchone()
                if result:
                    st.session_state.user = login_user.strip()
                    st.session_state.page = "Home"
                    st.rerun()
                else:
                    st.error("Incorrect username or password. Register first if needed.")
            except sqlite3.Error as e:
                st.error(f"Database error: {e}")
    st.stop()

user = st.session_state.user
with st.sidebar:
    st.title("🎀 AI Wardrobe")
    st.write(f"Hello, {user}")
    if st.button("Logout", use_container_width=True):
        st.session_state.user = None
        st.session_state.page = "Home"
        st.rerun()

page = st.session_state.page
if page == "Home":
    st.title("Welcome to Your Wardrobe 💗")
    st.write("Choose what you want to do.")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### ✨ Find Your Outfit")
        st.write("Discover combinations for your occasion.")
        if st.button("Find Your Outfit", key="home_find", use_container_width=True):
            go_to("Find Your Outfit")
            st.rerun()
    with c2:
        st.markdown("### 👗 My Wardrobe")
        st.write("Upload and view your clothing images.")
        if st.button("My Wardrobe", key="home_wardrobe", use_container_width=True):
            go_to("My Wardrobe")
            st.rerun()
    c3, c4 = st.columns(2)
    with c3:
        st.markdown("### 💌 Your Feedback")
        st.write("See the outfits you liked or disliked.")
        if st.button("Your Feedback", key="home_feedback", use_container_width=True):
            go_to("Your Feedback")
            st.rerun()
    with c4:
        st.markdown("### 🏠 Back to Home Page")
        st.write("Return to the main page.")
        if st.button("Back to Home Page", key="home_back", use_container_width=True):
            go_to("Home")
            st.rerun()

if page != "Home":
    if st.button("🏠 Back to Home Page", key="back_home"):
        go_to("Home")
        st.rerun()

if page == "My Wardrobe":
    st.title("👗 My Wardrobe")
    st.write("Upload individual clothing pictures or a PDF containing clothing pictures.")
    image_files = st.file_uploader("Choose clothing images", type=["jpg","jpeg","png","webp"],
                                   accept_multiple_files=True, key="multiple_clothes")
    pdf_files = st.file_uploader("Or upload a PDF of clothing images", type=["pdf"],
                                 accept_multiple_files=True, key="clothing_pdfs")
    pending = add_uploads_to_pending(image_files, pdf_files)
    if pending:
        st.subheader("Enter details for each clothing image 💗")
        st.caption("Check every extracted image and select the correct details before saving.")
        for i, (filename, data) in enumerate(pending):
            with st.expander(f"👗 {i + 1}. {filename}", expanded=True):
                st.image(data, width=150)
                st.text_input("Clothing name (optional)", value=os.path.splitext(filename)[0],
                              key=f"clothing_name_{i}")
                st.selectbox("Category", CATEGORY, key=f"category_{i}")
                st.selectbox("Type", TYPES, key=f"clothing_type_{i}")
                st.selectbox("Colour", COLORS, key=f"color_{i}")
                st.selectbox("Style", STYLES, key=f"style_{i}")
                st.multiselect("Suitable occasions", OCCASIONS, default=["Any"],
                               key=f"occasions_{i}")
        if st.button("💗 Save All Clothing", use_container_width=True):
            saved, failed = 0, []
            progress = st.progress(0)
            for i, (filename, data) in enumerate(pending):
                try:
                    image = Image.open(io.BytesIO(data)).convert("RGB")
                    category = st.session_state[f"category_{i}"]
                    clothing_type = st.session_state[f"clothing_type_{i}"]
                    color = st.session_state[f"color_{i}"]
                    style = st.session_state[f"style_{i}"]
                    name = st.session_state[f"clothing_name_{i}"].strip() or f"Clothing {i+1}"
                    occasions = st.session_state[f"occasions_{i}"] or ["Any"]
                    if "Any" in occasions:
                        occasions = ["Any"]
                    vector_text = ""
                    try:
                        with st.spinner(f"Extracting image features for {filename}..."):
                            vector_text = json.dumps(extract_clip(data))
                    except Exception as e:
                        st.warning(f"{filename} saved without CLIP features: {e}")
                    safe_user = hashlib.sha256(user.encode()).hexdigest()[:12]
                    out_name = f"{safe_user}_{os.urandom(5).hex()}.jpg"
                    image.save(os.path.join(IMG_DIR, out_name), format="JPEG")
                    con.execute("""INSERT INTO clothes
                        (username,name,category,type,color,style,occasions,image_path,clip_vector)
                        VALUES(?,?,?,?,?,?,?,?,?)""",
                        (user, name, category, clothing_type, color, style, ",".join(occasions),
                         os.path.join("wardrobe_images", out_name), vector_text))
                    con.commit()
                    saved += 1
                except Exception as e:
                    failed.append(f"{filename}: {e}")
                progress.progress((i + 1) / len(pending))
            if saved:
                st.success(f"Successfully saved {saved} clothing image(s)! 💗")
            for error in failed:
                st.error(f"Could not save: {error}")
            if not failed:
                st.rerun()

    clothes = con.execute("SELECT * FROM clothes WHERE username=? ORDER BY id DESC", (user,)).fetchall()
    st.subheader(f"My saved clothes ({len(clothes)})")
    if not clothes:
        st.info("Upload some clothing images to get started.")
    else:
        cols = st.columns(3)
        for i, item in enumerate(clothes):
            with cols[i % 3]:
                path = full_path(item["image_path"])
                if path and os.path.exists(path):
                    show_image(path)
                st.markdown(f"**{item['name']}**")
                st.caption(f"{item['category'] or 'Other'} · {item['type'] or 'Not specified'}")
                st.caption(f"Colour: {item['color'] or 'Unknown'}")
                st.caption(f"Style: {item['style'] or 'Any'}")
                st.caption(f"Occasions: {item['occasions'] or 'Any'}")
                if st.button("Delete", key=f"delete_{item['id']}_{i}", use_container_width=True):
                    if path and os.path.exists(path):
                        os.remove(path)
                    con.execute("DELETE FROM clothes WHERE id=? AND username=?", (item["id"], user))
                    con.commit()
                    st.rerun()

elif page == "Find Your Outfit":
    st.title("✨ Find Your Outfit")
    st.write("Choose an occasion to find combinations from your saved clothes.")
    clothes = con.execute("SELECT * FROM clothes WHERE username=?", (user,)).fetchall()
    if len(clothes) < 2:
        st.info("Open My Wardrobe and save at least two clothing items first.")
    else:
        c1, c2, c3 = st.columns(3)
        with c1:
            occasion = st.selectbox("Occasion", OCCASIONS)
        with c2:
            style_filter = st.selectbox("Style", STYLES)
        with c3:
            color_filter = st.selectbox("Preferred colour", ["Any"] + COLORS[:-1])
        eligible = []
        for item in clothes:
            allowed = [v.strip().lower() for v in (item["occasions"] or "Any").split(",")]
            if occasion != "Any" and "any" not in allowed and occasion.lower() not in allowed:
                continue
            if style_filter != "Any" and item["style"] not in (style_filter, "Any"):
                continue
            eligible.append(item)
        tops = [x for x in eligible if x["category"] == "Top"]
        bottoms = [x for x in eligible if x["category"] == "Bottom"]
        dresses = [x for x in eligible if x["category"] == "Dress"]
        extras = [x for x in eligible if x["category"] in ("Footwear", "Accessory")]
        pairs = [(a,b) for a in tops for b in bottoms]
        pairs += [(a,b) for a in dresses for b in extras]
        pairs += [(a,b) for a in tops for b in extras]
        pairs += [(a,b) for a in bottoms for b in extras]
        if not pairs:
            pairs = list(combinations(eligible, 2))
        unique = {}
        for a,b in pairs:
            if a["id"] != b["id"]:
                key = f"{min(a['id'],b['id'])}-{max(a['id'],b['id'])}"
                unique[key] = (a,b)
        pairs = list(unique.items())
        if not pairs:
            st.warning("No matching items. Try Occasion = Any or add more compatible clothes.")
        else:
            feedback_rows = con.execute("SELECT outfit_key,rating FROM feedback WHERE username=?", (user,)).fetchall()
            feedback_scores = {}
            for row in feedback_rows:
                feedback_scores[row["outfit_key"]] = feedback_scores.get(row["outfit_key"], 0) + (row["rating"] or 0)
            ranked = []
            for key, (a,b) in pairs:
                bonus = sum(1 for x in (a,b) if color_filter != "Any" and x["color"] == color_filter)
                sim = None
                try:
                    va = json.loads(a["clip_vector"]) if a["clip_vector"] else None
                    vb = json.loads(b["clip_vector"]) if b["clip_vector"] else None
                    if va and vb:
                        sim = similarity(va, vb)
                except Exception:
                    pass
                score = feedback_scores.get(key, 0) * 2 + bonus + (0.3 * sim if sim is not None else 0)
                ranked.append((score, key, a, b, sim))
            ranked.sort(key=lambda x: x[0], reverse=True)
            for number, (score, key, a, b, sim) in enumerate(ranked[:8], 1):
                st.markdown("---")
                st.subheader(f"Outfit idea {number}")
                left, right = st.columns(2)
                with left:
                    p = full_path(a["image_path"])
                    if p and os.path.exists(p):
                        show_image(p)
                    st.write(f"**{a['name']}** · {a['color']}")
                    st.caption(f"{a['category']} / {a['type']} / {a['style']}")
                with right:
                    p = full_path(b["image_path"])
                    if p and os.path.exists(p):
                        show_image(p)
                    st.write(f"**{b['name']}** · {b['color']}")
                    st.caption(f"{b['category']} / {b['type']} / {b['style']}")
                if sim is not None:
                    st.caption(f"CLIP image-vector similarity: {sim:.2f}")
                f1, f2 = st.columns(2)
                with f1:
                    if st.button("👍 Like", key=f"like_{key}_{number}", use_container_width=True):
                        con.execute("DELETE FROM feedback WHERE username=? AND outfit_key=?", (user, key))
                        con.execute("INSERT INTO feedback(username,outfit_key,rating) VALUES(?,?,?)", (user, key, 1))
                        con.commit()
                        st.rerun()
                with f2:
                    if st.button("👎 Dislike", key=f"dislike_{key}_{number}", use_container_width=True):
                        con.execute("DELETE FROM feedback WHERE username=? AND outfit_key=?", (user, key))
                        con.execute("INSERT INTO feedback(username,outfit_key,rating) VALUES(?,?,?)", (user, key, -1))
                        con.commit()
                        st.rerun()
    st.caption("CLIP extracts image features. Occasion/style filtering uses saved metadata; similarity is approximate.")

elif page == "Your Feedback":
    st.title("💌 Your Feedback")
    rows = con.execute("SELECT outfit_key,rating FROM feedback WHERE username=? ORDER BY id DESC", (user,)).fetchall()
    if not rows:
        st.info("No feedback yet. Like or dislike an outfit first.")
    else:
        names = {r["id"]: r["name"] for r in con.execute("SELECT id,name FROM clothes WHERE username=?", (user,)).fetchall()}
        st.metric("Feedback submitted", len(rows))
        for row in rows:
            label = "👍 Liked" if (row["rating"] or 0) > 0 else "👎 Disliked"
            try:
                ida, idb = [int(v) for v in row["outfit_key"].split("-")]
                outfit_text = f"{names.get(ida, 'Deleted item')} + {names.get(idb, 'Deleted item')}"
            except Exception:
                outfit_text = str(row["outfit_key"])
            st.write(f"{label} — {outfit_text}")

st.markdown("---")
st.caption("AI Wardrobe · CLIP + cosine similarity + rule-based filtering + feedback")
