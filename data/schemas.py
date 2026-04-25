from typing import Dict

SCHEMAS: Dict[str, dict] = {
    "GST Invoice": {
        "type": "object",
        "properties": {
            "gstin_seller": {"type": "string", "description": "15-character GSTIN of the seller"},
            "gstin_buyer": {"type": "string", "description": "15-character GSTIN of the buyer"},
            "invoice_number": {"type": "string", "description": "Unique invoice identifier"},
            "invoice_date": {"type": "string", "description": "Date of invoice in DD/MM/YYYY format"},
            "seller_name": {"type": "string", "description": "Name of the seller"},
            "buyer_name": {"type": "string", "description": "Name of the buyer"},
            "place_of_supply": {"type": "string", "description": "State of supply"},
            "hsn_code": {"type": "string", "description": "Harmonized System of Nomenclature code"},
            "item_description": {"type": "string", "description": "Description of the good or service"},
            "quantity": {"type": "number", "description": "Quantity of the item"},
            "unit_price": {"type": "number", "description": "Price per unit in INR"},
            "cgst_rate": {"type": "number", "description": "Central GST rate percentage"},
            "sgst_rate": {"type": "number", "description": "State GST rate percentage"},
            "igst_rate": {"type": "number", "description": "Integrated GST rate percentage"},
            "total_amount": {"type": "number", "description": "Total amount including taxes"},
            "tax_amount": {"type": "number", "description": "Total tax amount"},
        },
        "required": [
            "gstin_seller", "gstin_buyer", "invoice_number", "invoice_date",
            "seller_name", "buyer_name", "place_of_supply", "hsn_code",
            "item_description", "quantity", "unit_price", "total_amount"
        ],
        "additionalProperties": False
    },
    "PAN Card Application Form": {
        "type": "object",
        "properties": {
            "applicant_name": {"type": "string"},
            "father_name": {"type": "string"},
            "dob": {"type": "string", "description": "DD/MM/YYYY"},
            "gender": {"type": "string", "enum": ["M", "F", "T"]},
            "address_line1": {"type": "string"},
            "address_line2": {"type": "string"},
            "city": {"type": "string"},
            "state": {"type": "string"},
            "pincode": {"type": "string", "description": "6-digit PIN code"},
            "mobile": {"type": "string", "description": "10-digit mobile number"},
            "email": {"type": "string", "format": "email"},
            "existing_pan": {"type": "string", "description": "ABCDE1234F format"}
        },
        "required": [
            "applicant_name", "father_name", "dob", "gender", "address_line1",
            "city", "state", "pincode", "mobile"
        ],
        "additionalProperties": False
    },
    "FIR (First Information Report)": {
        "type": "object",
        "properties": {
            "fir_number": {"type": "string"},
            "police_station": {"type": "string"},
            "district": {"type": "string"},
            "state": {"type": "string"},
            "date_of_occurrence": {"type": "string", "description": "DD/MM/YYYY"},
            "time_of_occurrence": {"type": "string", "description": "HH:MM"},
            "complainant_name": {"type": "string"},
            "accused_name": {"type": "string"},
            "ipc_sections": {"type": "array", "items": {"type": "string"}},
            "incident_description": {"type": "string"},
            "investigating_officer": {"type": "string"}
        },
        "required": [
            "fir_number", "police_station", "district", "state", "date_of_occurrence",
            "complainant_name", "incident_description", "investigating_officer"
        ],
        "additionalProperties": False
    },
    "Medical Prescription": {
        "type": "object",
        "properties": {
            "patient_name": {"type": "string"},
            "patient_age": {"type": "integer"},
            "doctor_name": {"type": "string"},
            "doctor_reg_number": {"type": "string"},
            "prescription_date": {"type": "string", "description": "DD/MM/YYYY"},
            "diagnosis": {"type": "string"},
            "medications": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "drug_name": {"type": "string"},
                        "dosage": {"type": "string"},
                        "frequency": {"type": "string"},
                        "duration": {"type": "string"}
                    },
                    "required": ["drug_name", "dosage", "frequency", "duration"],
                    "additionalProperties": False
                }
            },
            "hospital_name": {"type": "string"},
            "follow_up_date": {"type": "string", "description": "DD/MM/YYYY"}
        },
        "required": [
            "patient_name", "patient_age", "doctor_name", "doctor_reg_number",
            "prescription_date", "medications"
        ],
        "additionalProperties": False
    },
    "Land Record (7/12 Utara)": {
        "type": "object",
        "properties": {
            "survey_number": {"type": "string"},
            "village": {"type": "string"},
            "taluka": {"type": "string"},
            "district": {"type": "string"},
            "state": {"type": "string"},
            "owner_name": {"type": "string"},
            "area_hectares": {"type": "number"},
            "land_type": {"type": "string"},
            "cultivation_season": {"type": "string"},
            "crop_name": {"type": "string"}
        },
        "required": [
            "survey_number", "village", "taluka", "district", "state",
            "owner_name", "area_hectares"
        ],
        "additionalProperties": False
    }
}
