import streamlit as st
import hashlib, io, json, uuid
import numpy as np
from PIL import Image
from itertools import combinations
from supabase import create_client
st.set_page_config(page_title="AI Wardrobe", page_icon="🎀", layout="wide")
try:
    @st.cache_resource
    def connect_supabase():
        return create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"])
    db = connect_supabase()
except Exception as e:
    st.error("Supabase connection failed. Check Streamlit Secrets.")
    st.code(str(e))
    st.stop()
BUCKET = "wardrobe-images"
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
def rows(table, username=None, order=None):
    q = db.table(table).select("*")
    if username is not None:
        q = q.eq("username", username)
    if order:
        q = q.order(order, desc=True)
    return q.execute().data or []
def password_hash(password):
    return hashlib.sha256(password.encode()).hexdigest()
def upload_image(data, username):
    image = Image.open(io.BytesIO(data)).convert("RGB")
    out = io.BytesIO()
    image.save(out, format="JPEG", quality=90)
    path = f"{hashlib.sha256(username.encode()).hexdigest()[:16]}/{uuid.uuid4().hex}.jpg"
    db.storage.from_(BUCKET).upload(
        path, out.getvalue(),
        file_options={"content-type": "image/jpeg", "upsert": "false"}
    )
    return path
def image_url(path):
    if not path:
        return None
    try:
        result = db.storage.from_(BUCKET).create_signed_url(path, 3600)
        url = result.get("signedURL") or result.get("signedUrl")
        if url and url.startswith("/"):
            url = st.secrets["SUPABASE_URL"].rstrip("/") + url
        return url
    except Exception:
        return None
def show_image(path, width=None):
    url = image_url(path)
    if url:
        st.image(url, width=width if width else "stretch")
    else:
        st.warning("Image unavailable. Check the Supabase Storage bucket.")
def delete_image(path):
    if path:
        db.storage.from_(BUCKET).remove([path])
@st.cache_resource(show_spinner="Loading CLIP AI model...")
def clip_model():
    from transformers import CLIPModel, CLIPProcessor
    import torch
    model_id = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(model_id)
    processor = CLIPProcessor.from_pretrained(model_id)
    model.eval()
    return model, processor, torch
def clip_features(data):
    model, processor, torch = clip_model()
    image = Image.open(io.BytesIO(data)).convert("RGB")
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
def pdf_images(pdf_data):
    import fitz
    doc, found_images, seen = fitz.open(stream=pdf_data, filetype="pdf"), [], set()
    for page_no, page in enumerate(doc):
        found = False
        for info in page.get_images(full=True):
            try:
                raw = doc.extract_image(info[0])["image"]
                im = Image.open(io.BytesIO(raw)).convert("RGB")
                out = io.BytesIO()
                im.save(out, format="JPEG")
                data = out.getvalue()
                digest = hashlib.sha256(data).hexdigest()
                if digest not in seen:
                    seen.add(digest)
                    found_images.append((f"Page {page_no + 1} image {len(found_images) + 1}.jpg", data))
                found = True
            except Exception:
                continue
        if not found:
            data = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False).tobytes("png")
            digest = hashlib.sha256(data).hexdigest()
            if digest not in seen:
                seen.add(digest)
                found_images.append((f"Page {page_no + 1}.png", data))
    doc.close()
    return found_images
def collect_uploads(images, pdfs):
    pending = []
    for f in images or []:
        try:
            data = f.getvalue()
            Image.open(io.BytesIO(data)).verify()
            pending.append((f.name, data))
        except Exception:
            st.warning(f"Could not read image: {f.name}")
    for f in pdfs or []:
        try:
            extracted = pdf_images(f.getvalue())
            if not extracted:
                st.warning(f"No images found in {f.name}.")
            else:
                st.caption(f"{f.name}: extracted {len(extracted)} image(s).")
                pending.extend((f"{f.name} — {name}", data) for name, data in extracted)
        except Exception as e:
            st.error(f"Could not read PDF {f.name}: {e}")
    return pending
st.markdown("""
<div class="hero"><h1>🎀 AI Wardrobe</h1>
<h3>Your personal virtual fashion assistant</h3>
<p>Discover outfit combinations from your own wardrobe.</p></div>
""", unsafe_allow_html=True)
if "user" not in st.session_state:
    st.session_state.user = None
if "page" not in st.session_state:
    st.session_state.page = "Home"
def go(page):
    st.session_state.page = page
if st.session_state.user is None:
    login_tab, register_tab = st.tabs(["Login", "Register"])
    with register_tab:
        st.subheader("Create your account 💗")
        with st.form("register_form"):
            new_user = st.text_input("Choose username")
            new_password = st.text_input("Choose password", type="password")
            confirm = st.text_input("Confirm password", type="password")
            register = st.form_submit_button("Register")
        if register:
            username = new_user.strip()
            if len(username) < 3:
                st.warning("Username needs at least 3 characters.")
            elif len(new_password) < 6:
                st.warning("Passwords need at least 6 characters.")
            elif new_password != confirm:
                st.warning("Passwords do not match.")
            else:
                try:
                    existing = db.table("users").select("username").eq("username", username).execute().data
                    if existing:
                        st.error("Username already exists. Please log in.")
                    else:
                        db.table("users").insert({
                            "username": username, "password": password_hash(new_password)
                        }).execute()
                        st.success("Account created! Now select the Login tab.")
                except Exception as e:
                    st.error(f"Registration failed: {e}")
    with login_tab:
        st.subheader("Welcome back 💕")
        with st.form("login_form"):
            login_user = st.text_input("Username")
            login_password = st.text_input("Password", type="password")
            login = st.form_submit_button("Login")
        if login:
            try:
                result = db.table("users").select("username").eq(
                    "username", login_user.strip()
                ).eq("password", password_hash(login_password)).execute().data
                if result:
                    st.session_state.user = login_user.strip()
                    st.session_state.page = "Home"
                    st.rerun()
                else:
                    st.error("Incorrect username or password. Register first if needed.")
            except Exception as e:
                st.error(f"Login failed: {e}")
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
            go("Find Your Outfit")
            st.rerun()
    with c2:
        st.markdown("### 👗 My Wardrobe")
        st.write("Upload and view your clothing images.")
        if st.button("My Wardrobe", key="home_wardrobe", use_container_width=True):
            go("My Wardrobe")
            st.rerun()
    c3, c4 = st.columns(2)
    with c3:
        st.markdown("### 💌 Your Feedback")
        st.write("See the outfits you liked or disliked.")
        if st.button("Your Feedback", key="home_feedback", use_container_width=True):
            go("Your Feedback")
            st.rerun()
    with c4:
        st.markdown("### 🏠 Back to Home Page")
        st.write("Return to the main page.")
        if st.button("Back to Home Page", key="home_back", use_container_width=True):
            go("Home")
            st.rerun()
if page != "Home" and st.button("🏠 Back to Home Page", key="back_home"):
    go("Home")
    st.rerun()
if page == "My Wardrobe":
    st.title("👗 My Wardrobe")
    st.write("Upload individual clothing pictures or a PDF containing clothing pictures.")
    image_files = st.file_uploader("Choose clothing images", type=["jpg", "jpeg", "png", "webp"],
                                   accept_multiple_files=True, key="multiple_clothes")
    pdf_files = st.file_uploader("Or upload a PDF of clothing images", type=["pdf"],
                                 accept_multiple_files=True, key="clothing_pdfs")
    pending = collect_uploads(image_files, pdf_files)
    if pending:
        st.subheader("Enter details for each clothing image 💗")
        st.caption("Check every extracted image and select the correct details before saving.")
        for i, (filename, data) in enumerate(pending):
            with st.expander(f"👗 {i + 1}. {filename}", expanded=True):
                st.image(data, width=150)
                st.text_input("Clothing name (optional)", value=filename.rsplit(".", 1)[0],
                              key=f"name_{i}")
                st.selectbox("Category", CATEGORY, key=f"category_{i}")
                st.selectbox("Type", TYPES, key=f"type_{i}")
                st.selectbox("Colour", COLORS, key=f"color_{i}")
                st.selectbox("Style", STYLES, key=f"style_{i}")
                st.multiselect("Suitable occasions", OCCASIONS, default=["Any"], key=f"occasion_{i}")
        if st.button("💗 Save All Clothing", use_container_width=True):
            saved, failed = 0, []
            progress = st.progress(0)
            for i, (filename, data) in enumerate(pending):
                path = None
                try:
                    category = st.session_state[f"category_{i}"]
                    clothing_type = st.session_state[f"type_{i}"]
                    color = st.session_state[f"color_{i}"]
                    style = st.session_state[f"style_{i}"]
                    name = st.session_state[f"name_{i}"].strip() or f"Clothing {i + 1}"
                    occasions = st.session_state[f"occasion_{i}"] or ["Any"]
                    if "Any" in occasions:
                        occasions = ["Any"]
                    vector = ""
                    try:
                        with st.spinner(f"Extracting image features for {filename}..."):
                            vector = json.dumps(clip_features(data))
                    except Exception as e:
                        st.warning(f"{filename} saved without CLIP features: {e}")
                    path = upload_image(data, user)
                    db.table("clothes").insert({
                        "username": user, "name": name, "category": category,
                        "type": clothing_type, "color": color, "style": style,
                        "occasions": ",".join(occasions), "image_path": path,
                        "clip_vector": vector
                    }).execute()
                    saved += 1
                except Exception as e:
                    failed.append(f"{filename}: {e}")
                    if path:
                        try:
                            delete_image(path)
                        except Exception:
                            pass
                progress.progress((i + 1) / len(pending))
            if saved:
                st.success(f"Successfully saved {saved} clothing image(s)! 💗")
            for error in failed:
                st.error(f"Could not save: {error}")
            if saved:
                st.rerun()
    try:
        clothes = rows("clothes", user, "id")
    except Exception as e:
        st.error(f"Could not load wardrobe: {e}")
        clothes = []
    st.subheader(f"My saved clothes ({len(clothes)})")
    if not clothes:
        st.info("Upload some clothing images to get started.")
    else:
        cols = st.columns(3)
        for i, item in enumerate(clothes):
            with cols[i % 3]:
                show_image(item.get("image_path"))
                st.markdown(f"**{item.get('name') or 'Clothing'}**")
                st.caption(f"{item.get('category') or 'Other'} · {item.get('type') or 'Not specified'}")
                st.caption(f"Colour: {item.get('color') or 'Unknown'}")
                st.caption(f"Style: {item.get('style') or 'Any'}")
                st.caption(f"Occasions: {item.get('occasions') or 'Any'}")
                if st.button("Delete", key=f"delete_{item['id']}_{i}", use_container_width=True):
                    try:
                        delete_image(item.get("image_path"))
                        db.table("clothes").delete().eq("id", item["id"]).eq("username", user).execute()
                        st.rerun()
                    except Exception as e:
                        st.error(f"Delete failed: {e}")
elif page == "Find Your Outfit":
    st.title("✨ Find Your Outfit")
    st.write("Choose an occasion to find combinations from your saved clothes.")
    try:
        clothes = rows("clothes", user)
    except Exception as e:
        st.error(f"Could not load clothing: {e}")
        clothes = []
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
            allowed = [v.strip().lower() for v in (item.get("occasions") or "Any").split(",")]
            if occasion != "Any" and "any" not in allowed and occasion.lower() not in allowed:
                continue
            if style_filter != "Any" and item.get("style") not in (style_filter, "Any"):
                continue
            eligible.append(item)
        tops = [x for x in eligible if x.get("category") == "Top"]
        bottoms = [x for x in eligible if x.get("category") == "Bottom"]
        dresses = [x for x in eligible if x.get("category") == "Dress"]
        extras = [x for x in eligible if x.get("category") in ("Footwear", "Accessory")]
        pairs = ([(a, b) for a in tops for b in bottoms] +
                 [(a, b) for a in dresses for b in extras] +
                 [(a, b) for a in tops for b in extras] +
                 [(a, b) for a in bottoms for b in extras])
        if not pairs:
            pairs = list(combinations(eligible, 2))
        unique = {}
        for a, b in pairs:
            if a["id"] != b["id"]:
                key = f"{min(a['id'], b['id'])}-{max(a['id'], b['id'])}"
                unique[key] = (a, b)
        pairs = list(unique.items())
        if not pairs:
            st.warning("No matching items. Try Occasion = Any or add more compatible clothes.")
        else:
            try:
                feedback = rows("feedback", user)
            except Exception as e:
                st.error(f"Could not load feedback: {e}")
                feedback = []
            scores = {}
            for row in feedback:
                key = row.get("outfit_key")
                scores[key] = scores.get(key, 0) + (row.get("rating") or 0)
            ranked = []
            for key, (a, b) in pairs:
                color_bonus = sum(1 for x in (a, b)
                                  if color_filter != "Any" and x.get("color") == color_filter)
                sim = None
                try:
                    va = json.loads(a["clip_vector"]) if a.get("clip_vector") else None
                    vb = json.loads(b["clip_vector"]) if b.get("clip_vector") else None
                    if va and vb:
                        sim = similarity(va, vb)
                except Exception:
                    pass
                score = scores.get(key, 0) * 2 + color_bonus + (0.3 * sim if sim is not None else 0)
                ranked.append((score, key, a, b, sim))
            ranked.sort(key=lambda x: x[0], reverse=True)
            for number, (_, key, a, b, sim) in enumerate(ranked[:8], 1):
                st.markdown("---")
                st.subheader(f"Outfit idea {number}")
                left, right = st.columns(2)
                with left:
                    show_image(a.get("image_path"))
                    st.write(f"**{a.get('name')}** · {a.get('color')}")
                    st.caption(f"{a.get('category')} / {a.get('type')} / {a.get('style')}")
                with right:
                    show_image(b.get("image_path"))
                    st.write(f"**{b.get('name')}** · {b.get('color')}")
                    st.caption(f"{b.get('category')} / {b.get('type')} / {b.get('style')}")
                if sim is not None:
                    st.caption(f"CLIP image-vector similarity: {sim:.2f}")
                like_col, dislike_col = st.columns(2)
                for col, label, rating in [(like_col, "👍 Like", 1), (dislike_col, "👎 Dislike", -1)]:
                    with col:
                        if st.button(label, key=f"feedback_{rating}_{key}_{number}", use_container_width=True):
                            try:
                                db.table("feedback").delete().eq("username", user).eq("outfit_key", key).execute()
                                db.table("feedback").insert({
                                    "username": user, "outfit_key": key, "rating": rating
                                }).execute()
                                st.rerun()
                            except Exception as e:
                                st.error(f"Could not save feedback: {e}")
    st.caption("CLIP extracts image features. Occasion/style filtering uses saved metadata; similarity is approximate.")
elif page == "Your Feedback":
    st.title("💌 Your Feedback")
    try:
        feedback = rows("feedback", user, "id")
        clothes = rows("clothes", user)
    except Exception as e:
        st.error(f"Could not load feedback: {e}")
        feedback, clothes = [], []
    if not feedback:
        st.info("No feedback yet. Like or dislike an outfit first.")
    else:
        names = {r["id"]: r.get("name", "Clothing") for r in clothes}
        st.metric("Feedback submitted", len(feedback))
        for row in feedback:
            label = "👍 Liked" if (row.get("rating") or 0) > 0 else "👎 Disliked"
            try:
                ida, idb = [int(v) for v in row["outfit_key"].split("-")]
                text = f"{names.get(ida, 'Deleted item')} + {names.get(idb, 'Deleted item')}"
            except Exception:
                text = str(row.get("outfit_key", ""))
            st.write(f"{label} — {text}")
st.markdown("---")
st.caption("AI Wardrobe · CLIP + cosine similarity + rule-based filtering + feedback")
