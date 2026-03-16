import streamlit as st
import numpy as np
from ultralytics import YOLO
from PIL import Image
from supabase import create_client
import uuid
import io
import requests

# EMAIL
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# =====================================================
# CONFIG
# =====================================================

st.set_page_config(page_title="Lost&Found", layout="wide")

SUPABASE_URL = st.secrets["supabase"]["url"]
SUPABASE_KEY = st.secrets["supabase"]["key"]
ADMIN_PASSWORD = st.secrets["admin_password"]

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# =====================================================
# SESSION STATE
# =====================================================

if "page" not in st.session_state:
    st.session_state.page = "Galerie"

if "admin_logged_in" not in st.session_state:
    st.session_state.admin_logged_in = False

if "batch_size" not in st.session_state:
    st.session_state.batch_size = 12

params = st.query_params
if "page" in params:
    st.session_state.page = params["page"]

# =====================================================
# LOAD YOLO MODEL
# =====================================================

@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# =====================================================
# EMAIL
# =====================================================

def send_email(entry):

    recipient = entry.get("email", "")

    if not recipient:
        st.warning("Keine Email angegeben")
        return

    sender_email = st.secrets["email"]["address"]
    sender_password = st.secrets["email"]["password"]

    subject = "Lost&Found: Ihr Fundstück wurde gefunden!"

    message_text = f"""
Hallo,

Ihr Fundstück wurde gefunden!

Bild:
{entry["image_url"]}

Viele Grüße
Lost&Found
"""

    msg = MIMEMultipart()

    msg["From"] = sender_email
    msg["To"] = recipient
    msg["Subject"] = subject

    msg.attach(MIMEText(message_text, "plain"))

    try:

        with smtplib.SMTP("smtp.gmail.com", 587) as server:

            server.starttls()

            server.login(sender_email, sender_password)

            server.sendmail(
                sender_email,
                recipient,
                msg.as_string()
            )

        st.success("Email gesendet")

    except Exception as e:

        st.error(e)

# =====================================================
# YOLO DETECTION
# =====================================================

def detect_objects(image):

    results = model(image)

    result = results[0]

    names = result.names

    classes = result.boxes.cls.tolist()

    objects = list(set([names[int(c)] for c in classes]))

    confidence = float(np.max(result.boxes.conf.tolist())) if len(result.boxes.conf) else 0

    return objects, confidence

# =====================================================
# STORAGE
# =====================================================

def upload_image(image, label):

    filename = f"{label}/{uuid.uuid4()}.jpg"

    buffer = io.BytesIO()

    image.save(buffer, format="JPEG")

    buffer.seek(0)

    supabase.storage.from_("fundbilder").upload(
        filename,
        buffer.getvalue(),
        {"content-type": "image/jpeg"}
    )

    url = supabase.storage.from_("fundbilder").get_public_url(filename)

    return url

# =====================================================
# SAVE METADATA
# =====================================================

def save_metadata(url, objects, confidence, tag, status, description, email):

    object_string = ", ".join(objects)

    data = {

        "image_url": url,
        "predicted_class": object_string,
        "confidence": confidence,
        "tag": tag,
        "status": status,
        "description": description,
        "email": email
    }

    supabase.table("fundstuecke").insert(data).execute()

# =====================================================
# LOAD ENTRIES
# =====================================================

def load_entries(search=None, tag=None, status=None):

    query = supabase.table("fundstuecke").select("*").order("created_at", desc=True)

    if search:
        query = query.ilike("predicted_class", f"%{search}%")

    if tag and tag != "Alle":
        query = query.eq("tag", tag)

    if status and status != "Alle":
        query = query.eq("status", status)

    res = query.execute()

    return res.data

# =====================================================
# DELETE
# =====================================================

def delete_entry(entry):

    image_url = entry["image_url"]

    path = image_url.split("/fundbilder/")[1]

    supabase.storage.from_("fundbilder").remove([path])

    supabase.table("fundstuecke").delete().eq("id", entry["id"]).execute()

# =====================================================
# GALLERY
# =====================================================

def render_gallery(entries, admin=False):

    cols = st.columns(4)

    for i, entry in enumerate(entries):

        with cols[i % 4]:

            try:

                response = requests.get(entry["image_url"])

                image = Image.open(io.BytesIO(response.content))

                st.image(image, width="stretch")

            except:

                st.warning("Bild Fehler")

            tags = f"""
**Objekte:** {entry["predicted_class"]}

**Farbe:** {entry.get("tag","")}

**Status:** {entry.get("status","")}
"""

            with st.expander("Details"):

                st.markdown(tags)

                st.write(entry.get("description",""))

                st.write(entry.get("email",""))

                if admin:

                    if entry["status"] == "Missing":

                        if st.button("Email", key=f"mail{entry['id']}"):

                            send_email(entry)

                    if st.button("Delete", key=f"del{entry['id']}"):

                        delete_entry(entry)

                        st.rerun()

# =====================================================
# PAGE ROUTER
# =====================================================

page = st.session_state.page

# =====================================================
# GALERIE
# =====================================================

if page == "Galerie":

    st.title("Lost&Found")

    search = st.text_input("Objekt suchen (z.B. shoe, backpack, jacket)")

    c1,c2 = st.columns(2)

    with c1:
        tag = st.selectbox("Farbe", ["Alle","rot","blau","grün","gelb","schwarz","weiß"])

    with c2:
        status = st.selectbox("Status", ["Alle","Found","Missing"])

    entries = load_entries(search, tag, status)

    entries = entries[:st.session_state.batch_size]

    render_gallery(entries)

    if len(load_entries()) > st.session_state.batch_size:

        if st.button("Mehr laden"):

            st.session_state.batch_size += 12

            st.rerun()

# =====================================================
# UPLOAD
# =====================================================

if page == "Upload":

    st.title("Neues Fundstück")

    uploaded = st.file_uploader("Bild", type=["jpg","png","jpeg"])

    camera = st.camera_input("oder Foto")

    image_file = uploaded if uploaded else camera

    if image_file:

        image = Image.open(image_file).convert("RGB")

        st.image(image, width="stretch")

        objects, confidence = detect_objects(image)

        st.subheader("Erkannte Objekte")

        st.write(objects)

        st.progress(confidence)

        tag = st.selectbox("Farbe", ["rot","blau","grün","gelb","schwarz","weiß"])

        status = st.selectbox("Status", ["Found","Missing"])

        description = st.text_area("Beschreibung")

        email = st.text_input("Email optional")

        if st.button("Speichern"):

            label = objects[0] if objects else "unknown"

            url = upload_image(image, label)

            save_metadata(
                url,
                objects,
                confidence,
                tag,
                status,
                description,
                email
            )

            st.success("Gespeichert")

# =====================================================
# ADMIN
# =====================================================

if page == "Admin":

    st.title("Admin")

    if not st.session_state.admin_logged_in:

        pw = st.text_input("Passwort", type="password")

        if st.button("Login"):

            if pw == ADMIN_PASSWORD:

                st.session_state.admin_logged_in = True

                st.rerun()

            else:

                st.error("Falsch")

    else:

        if st.button("Logout"):

            st.session_state.admin_logged_in = False

            st.rerun()

        search = st.text_input("Objekt suchen")

        entries = load_entries(search)

        render_gallery(entries, admin=True)