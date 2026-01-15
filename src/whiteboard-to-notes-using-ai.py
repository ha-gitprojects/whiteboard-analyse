import base64
import json
from datetime import datetime
from pathlib import Path

import gradio as gr
from anthropic import Anthropic


# ============================================================
# Projekt: Classroom Whiteboard to Structured Notes
# Zweck: Didaktische Aufbereitung von Vorlesungstafeln
# ============================================================


# ------------------------------------------------------------
# Hilfsfunktionen
# ------------------------------------------------------------

def bild_als_base64(pfad: str) -> str:
    """
    Liest ein Bild von der Festplatte und kodiert es
    für die Übergabe an ein multimodales KI-Modell.
    """
    with open(pfad, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def zeitstempel() -> str:
    """Erzeugt einen einfachen Zeitstempel für Dateinamen"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def json_speichern(daten: dict) -> str:
    """
    Speichert das Ergebnis lokal als JSON-Datei.
    Dient nur der Demonstration.
    """
    out_dir = Path("whiteboard_exports")
    out_dir.mkdir(exist_ok=True)

    dateiname = f"lecture_notes_{zeitstempel()}.json"
    pfad = out_dir / dateiname

    with open(pfad, "w", encoding="utf-8") as f:
        json.dump(daten, f, indent=4, ensure_ascii=False)

    return str(pfad)


# ------------------------------------------------------------
# KI-Service: Whiteboard-Analyse mit Claude
# ------------------------------------------------------------

class WhiteboardAssistent:
    """
    Kapselt die gesamte Kommunikation mit Claude.
    Fokus: Erkennen, Strukturieren, Didaktisieren.
    """

    def __init__(self):
        self.client = Anthropic()

    def analysiere_whiteboard(self, bild_base64: str) -> dict:
        """
        Sendet ein Whiteboard-Bild an Claude und erwartet
        strukturierte Lernnotizen als JSON.
        """

        system_prompt = """
        Du bist ein didaktischer KI-Assistent für Hochschullehre.

        Deine Aufgabe:
        - Erkenne Inhalte auf einem Vorlesungs-Whiteboard
        - Identifiziere Themen, Text, mathematische Formeln und Diagramme
        - Strukturiere alles logisch und lernfreundlich
        - Antworte ausschließlich mit validem JSON
        """

        ziel_schema = {
            "topic": "string",
            "sections": [
                {
                    "title": "string",
                    "content_type": "text | formula | diagram",
                    "content": "string",
                    "key_points": ["string"]
                }
            ],
            "flashcards": [
                {
                    "question": "string",
                    "answer": "string"
                }
            ],
            "summary": "string"
        }

        message = self.client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=3000,
            system=system_prompt,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Analysiere dieses Whiteboard-Foto aus einer Vorlesung "
                                "und strukturiere den Inhalt gemäß folgendem JSON-Schema:\n\n"
                                f"{json.dumps(ziel_schema, indent=4)}"
                            )
                        },
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": bild_base64
                            }
                        }
                    ]
                }
            ]
        )

        rohtext = message.content[0].text
        bereinigt = rohtext.replace("```json", "").replace("```", "").strip()
        return json.loads(bereinigt)


# ------------------------------------------------------------
# Gradio Interface
# ------------------------------------------------------------

def verarbeite_whiteboard(bild_datei):
    """
    Zentrale Callback-Funktion für Gradio.
    """
    assistent = WhiteboardAssistent()
    b64 = bild_als_base64(bild_datei)
    ergebnis = assistent.analysiere_whiteboard(b64)
    speicherort = json_speichern(ergebnis)

    return ergebnis, f"Gespeichert unter: {speicherort}"


with gr.Blocks(title="Classroom Whiteboard to Structured Notes") as demo:
    gr.Markdown(
        """
        # Classroom Whiteboard → Structured Notes

        Dieses Tool demonstriert, wie KI Vorlesungstafeln analysiert
        und daraus strukturierte Lernmaterialien und Lernkarten erzeugt.
        """
    )

    bild_input = gr.Image(
        type="filepath",
        label="Whiteboard-Foto hochladen"
    )

    analyse_button = gr.Button("Whiteboard analysieren")

    json_output = gr.JSON(label="Strukturierte Lernnotizen")
    status_text = gr.Textbox(label="Status")

    analyse_button.click(
        fn=verarbeite_whiteboard,
        inputs=bild_input,
        outputs=[json_output, status_text]
    )


if __name__ == "__main__":
    demo.launch()
