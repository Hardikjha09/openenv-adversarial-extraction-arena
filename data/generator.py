import json
import random
import uuid
import argparse
from typing import Dict, Any, List
from faker import Faker
from data.schemas import SCHEMAS

# Initialize faker with Indian locale
fake = Faker('en_IN')

def generate_gstin() -> str:
    """Generate fake GSTIN: 2 digits + 5 uppercase letters + 4 digits + 1 letter + 1 alphanumeric + Z + 1 alphanumeric"""
    return f"{random.randint(10, 99)}{fake.lexify('?????', letters='ABCDEFGHIJKLMNOPQRSTUVWXYZ')}{random.randint(1000, 9999)}{fake.lexify('?', letters='ABCDEFGHIJKLMNOPQRSTUVWXYZ')}{fake.lexify('?', letters='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')}Z{fake.lexify('?', letters='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')}"

def generate_pan() -> str:
    """Generate fake PAN: ABCDE1234F"""
    return f"{fake.lexify('?????', letters='ABCDEFGHIJKLMNOPQRSTUVWXYZ')}{random.randint(1000, 9999)}{fake.lexify('?', letters='ABCDEFGHIJKLMNOPQRSTUVWXYZ')}"

def generate_date() -> str:
    return fake.date_between(start_date='-5y', end_date='today').strftime("%d/%m/%Y")

def generate_gst_invoice() -> Dict[str, Any]:
    gold = {
        "gstin_seller": generate_gstin(),
        "gstin_buyer": generate_gstin(),
        "invoice_number": f"INV/{fake.year()}/{random.randint(1000, 9999)}",
        "invoice_date": generate_date(),
        "seller_name": fake.company(),
        "buyer_name": fake.company(),
        "place_of_supply": fake.state(),
        "hsn_code": str(random.randint(1000, 99999999)),
        "item_description": fake.catch_phrase(),
        "quantity": float(random.randint(1, 100)),
        "unit_price": round(random.uniform(10.0, 5000.0), 2),
        "cgst_rate": 9.0,
        "sgst_rate": 9.0,
        "igst_rate": 0.0,
    }
    gold["total_amount"] = round(gold["quantity"] * gold["unit_price"], 2)
    tax_rate = gold["cgst_rate"] + gold["sgst_rate"] + gold["igst_rate"]
    gold["tax_amount"] = round(gold["total_amount"] * (tax_rate / 100), 2)
    gold["total_amount"] += gold["tax_amount"]

    text = f"""TAX INVOICE
==================================================
Seller: {gold['seller_name']}
GSTIN: {gold['gstin_seller']}

Buyer: {gold['buyer_name']}
GSTIN: {gold['gstin_buyer']}
Place of Supply: {gold['place_of_supply']}

Invoice No: {gold['invoice_number']}
Date: {gold['invoice_date']}
--------------------------------------------------
Item Details:
Description: {gold['item_description']}
HSN Code: {gold['hsn_code']}
Quantity: {gold['quantity']}
Unit Price: ₹{gold['unit_price']:.2f}

Taxes:
CGST Rate: {gold['cgst_rate']}%
SGST Rate: {gold['sgst_rate']}%
IGST Rate: {gold['igst_rate']}%
Tax Amount: ₹{gold['tax_amount']:.2f}

--------------------------------------------------
Total Amount: ₹{gold['total_amount']:.2f}
=================================================="""

    return {
        "id": str(uuid.uuid4()),
        "type": "GST Invoice",
        "text": text,
        "gold": gold,
        "schema": SCHEMAS["GST Invoice"]
    }

def generate_pan_application() -> Dict[str, Any]:
    gold = {
        "applicant_name": fake.name(),
        "father_name": fake.name_male(),
        "dob": generate_date(),
        "gender": random.choice(["M", "F"]),
        "address_line1": fake.building_number() + " " + fake.street_name(),
        "address_line2": fake.street_address(),
        "city": fake.city(),
        "state": fake.state(),
        "pincode": fake.postcode(),
        "mobile": fake.numerify("##########"),
        "email": fake.email()
    }
    if random.random() > 0.5:
        gold["existing_pan"] = generate_pan()

    text = f"""FORM 49A
Application for Allotment of Permanent Account Number
------------------------------------------------------
1. Full Name: {gold['applicant_name']}
2. Father's Name: {gold['father_name']}
3. Date of Birth: {gold['dob']}
4. Gender: {'Male' if gold['gender'] == 'M' else 'Female'}

5. Address for Communication:
   Line 1: {gold['address_line1']}
   Line 2: {gold['address_line2']}
   Town/City/District: {gold['city']}
   State: {gold['state']}
   PIN Code: {gold['pincode']}

6. Contact Details:
   Mobile No.: +91 {gold['mobile']}
   Email ID: {gold['email']}
"""
    if "existing_pan" in gold:
        text += f"\n7. Existing PAN (if any): {gold['existing_pan']}\n"

    return {
        "id": str(uuid.uuid4()),
        "type": "PAN Card Application Form",
        "text": text,
        "gold": gold,
        "schema": SCHEMAS["PAN Card Application Form"]
    }

def generate_fir() -> Dict[str, Any]:
    gold = {
        "fir_number": f"FIR/{fake.year()}/{random.randint(10, 999)}",
        "police_station": f"{fake.city()} PS",
        "district": fake.city(),
        "state": fake.state(),
        "date_of_occurrence": generate_date(),
        "time_of_occurrence": fake.time("%H:%M"),
        "complainant_name": fake.name(),
        "accused_name": fake.name() if random.random() > 0.2 else "Unknown",
        "ipc_sections": [str(random.randint(100, 500)), str(random.randint(100, 500))],
        "incident_description": fake.text(max_nb_chars=200),
        "investigating_officer": fake.name()
    }

    text = f"""FIRST INFORMATION REPORT (Under Section 154 Cr.P.C.)
---------------------------------------------------
1. District: {gold['district']}
   State: {gold['state']}
   Police Station: {gold['police_station']}
   FIR No: {gold['fir_number']}

2. Date of Occurrence: {gold['date_of_occurrence']}
   Time of Occurrence: {gold['time_of_occurrence']}

3. Complainant Name: {gold['complainant_name']}
4. Accused Details: {gold['accused_name']}

5. Sections Applied: {', '.join(gold['ipc_sections'])}

6. Incident Description:
   {gold['incident_description']}

7. Investigating Officer: {gold['investigating_officer']}
---------------------------------------------------"""

    return {
        "id": str(uuid.uuid4()),
        "type": "FIR (First Information Report)",
        "text": text,
        "gold": gold,
        "schema": SCHEMAS["FIR (First Information Report)"]
    }

def generate_medical_prescription() -> Dict[str, Any]:
    meds = []
    for _ in range(random.randint(1, 4)):
        meds.append({
            "drug_name": random.choice(["Paracetamol", "Amoxicillin", "Ibuprofen", "Cetirizine", "Metformin", "Atorvastatin"]),
            "dosage": random.choice(["500mg", "250mg", "10mg", "5mg"]),
            "frequency": random.choice(["1-0-1", "1-1-1", "0-0-1", "SOS"]),
            "duration": f"{random.randint(3, 14)} days"
        })

    gold = {
        "patient_name": fake.name(),
        "patient_age": random.randint(5, 85),
        "doctor_name": f"Dr. {fake.name()}",
        "doctor_reg_number": f"{fake.state()[:2].upper()}/{random.randint(1000, 99999)}",
        "prescription_date": generate_date(),
        "diagnosis": fake.sentence(nb_words=4),
        "medications": meds,
        "hospital_name": f"{fake.last_name()} Clinic",
        "follow_up_date": fake.date_between(start_date='+7d', end_date='+30d').strftime("%d/%m/%Y")
    }

    med_text = "\n".join([f"   - {m['drug_name']} {m['dosage']} | Freq: {m['frequency']} | For: {m['duration']}" for m in meds])

    text = f"""=============================================
{gold['hospital_name']}
Dr. Name: {gold['doctor_name']}
Reg No: {gold['doctor_reg_number']}
=============================================
Date: {gold['prescription_date']}

Patient Details:
Name: {gold['patient_name']}
Age: {gold['patient_age']} Years

Diagnosis:
{gold['diagnosis']}

Rx:
{med_text}

Next Visit: {gold['follow_up_date']}
============================================="""

    return {
        "id": str(uuid.uuid4()),
        "type": "Medical Prescription",
        "text": text,
        "gold": gold,
        "schema": SCHEMAS["Medical Prescription"]
    }

def generate_land_record() -> Dict[str, Any]:
    gold = {
        "survey_number": f"{random.randint(1, 999)}/{random.randint(1, 9)}",
        "village": fake.city(),
        "taluka": fake.city(),
        "district": fake.city(),
        "state": fake.state(),
        "owner_name": fake.name(),
        "area_hectares": round(random.uniform(0.5, 15.0), 2),
        "land_type": random.choice(["Irrigated", "Dry", "Non-Agricultural"]),
        "cultivation_season": random.choice(["Kharif", "Rabi", "Zaid"]),
        "crop_name": random.choice(["Wheat", "Rice", "Sugarcane", "Cotton", "Jowar"])
    }

    text = f"""EXTRACT OF 7/12 (LAND RECORD)
----------------------------------------
State: {gold['state']}
District: {gold['district']}
Taluka: {gold['taluka']}
Village: {gold['village']}

Survey / Gat No.: {gold['survey_number']}

Area Details:
Total Area: {gold['area_hectares']} Hectares
Land Type: {gold['land_type']}

Occupant Details:
Name of the Occupant / Owner: {gold['owner_name']}

Crop Details:
Season: {gold['cultivation_season']}
Crop Sown: {gold['crop_name']}
----------------------------------------"""

    return {
        "id": str(uuid.uuid4()),
        "type": "Land Record (7/12 Utara)",
        "text": text,
        "gold": gold,
        "schema": SCHEMAS["Land Record (7/12 Utara)"]
    }

def generate_corpus(n: int = 1500, seed: int = 42, output_file: str = "data/corpus.json"):
    random.seed(seed)
    Faker.seed(seed)
    
    generators = [
        generate_gst_invoice,
        generate_pan_application,
        generate_fir,
        generate_medical_prescription,
        generate_land_record
    ]
    
    documents = []
    
    # 60% train, 40% holdout based on requested 900/600 split
    split_point = int(n * 0.6)
    
    for i in range(n):
        generator_func = random.choice(generators)
        doc = generator_func()
        doc["split"] = "train" if i < split_point else "holdout"
        documents.append(doc)
        
        if (i + 1) % 100 == 0:
            print(f"Generated {i + 1}/{n} documents...")
            
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(documents, f, indent=2, ensure_ascii=False)
    
    print(f"Successfully saved {len(documents)} documents to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=1500, help="Number of documents to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default="data/corpus.json", help="Output JSON file")
    args = parser.parse_args()
    
    generate_corpus(n=args.n, seed=args.seed, output_file=args.output)
