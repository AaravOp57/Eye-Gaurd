from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak

path = "/mnt/data/EyeGuard_AI_Complete_Log_Book_Updated.pdf"

doc = SimpleDocTemplate(
    path,
    pagesize=A4,
    rightMargin=12*mm,
    leftMargin=12*mm,
    topMargin=12*mm,
    bottomMargin=12*mm
)

styles = getSampleStyleSheet()
title = ParagraphStyle(
    "LogTitle", parent=styles["Title"], alignment=TA_CENTER,
    fontSize=18, leading=22, spaceAfter=8
)
date_style = ParagraphStyle(
    "Date", parent=styles["Normal"], fontSize=8.2, leading=10
)
body_style = ParagraphStyle(
    "Body", parent=styles["Normal"], fontSize=8.2, leading=10.5
)
small = ParagraphStyle(
    "Small", parent=styles["Normal"], fontSize=7.5, leading=9
)

entries = [
("20th May 2026",
"Our group was formed, and we had our first discussion about the project. We shared a few initial ideas and discussed different topics. During the discussion, we also decided that we would design and develop an app for our project. At the end of the meeting, everyone agreed to do some research individually and share their ideas with the group so that we could select the best topic together."),
("10th June 2026",
"All the group members shared the project ideas they had researched. We discussed different topics, such as <b><i>AI OcuShield for Spinal Posture Correction, AI Tree Guardian, Green Energy and Renewable Innovation, AI Hydration Shield, and the UV Self-Purifying Water Bottle</i></b>. Along with discussing these ideas, we also checked whether they were practically possible and whether we would be able to design and develop the app successfully. We decided that the final topic would be selected only if it was practical, useful, and could be developed by our team."),
("14th June 2026",
"After discussing all the ideas, our <b><i>group decided to select AI OcuShield as our final project topic</i></b>. We planned to design an AI-based app that helps users maintain the correct sitting posture while working on a computer. The <b><i>main aim of the app is to reduce eye strain and improve spinal posture by giving timely alerts and guidance</i></b>. We also understood that designing this app would be our biggest challenge. Along with this, we decided to <b><i>prepare a questionnaire to conduct a survey</i></b> and find out whether people would find the app useful and if they would be interested in using it."),
("15th June 2026",
"We started planning the development of our app and researched the problem of eye strain caused by long screen time. We <b><i>explored Google MediaPipe Face Mesh for face tracking and OpenCV for webcam</i></b> access to understand how these technologies can be used in our project."),
("17th June 2026",
"Our group discussed the main features of the app and <b><i>decided to include blink detection, screen detection monitoring, voice alerts, and warning messages</i></b>. We also finalized the tools required for development, such as <b><i>Python, NumPy, and pyttx3, along with MediaPipe and OpenCV</i></b>."),
("19th June 2026",
"We starting developing the app by implementing the <b><i>webcam feed, face tracking, and blink detection</i></b>. We <b><i>tested the app</i></b> in different conditions, <b><i>fixed a few errors, and improved its accuracy</i></b> so that it could provide better alerts and reminders."),
("26th June 2026",
"Our group discussed and <b><i>finalized the questionnaire</i></b> questions for the survey. We also reviewed the app's functionality and discussed how efficiently its features were working. We discussed how to <b><i>write the abstract, results, and conclusion</i></b> for our project. After that, we decided to start <b><i>making the Google Presentation</i></b> and organized its content. We also <b><i>divided the project work among the group members</i></b> and decided who would be responsible for each part of the project file."),
("30th June 2026",
"We <b><i>created our questionnaire in Microsoft Forms and circulated</i></b> it online among different groups, friends, and family members to collect responses and feedback about our project."),
("4th July 2026",
"Our group discussed the preparation for the upcoming project presentation. We <b><i>reviewed what should be included</i></b> in our PPT, including the log sheet, survey details, and all the work completed so far. We decided to wait for more survey responses till the night and analyse the results the next day. We also planned to take printouts of the complete project file and carry a laptop on the presentation day to demonstrate our app."),
("6th July 2026",
"Our group analysed the survey responses and updated the survey results in our Google Presentation. We also completed and updated the remaining sections of the project, including the Procedure and Materials Required. Finally, we discussed taking printouts of the completed project file and getting it bound before the presentation."),
("28th July 2026",
"After our project was selected for the first round, our group discussed how to make the app more useful and interactive. We planned to add an Eye Health Score so that users could easily understand the condition of their eye health based on their screen habits. We also discussed a few new ideas to improve the overall user experience."),
("3rd August 2026",
"Our group finalized the new features for the app. We decided that the app would provide an Eye Health Score, along with smart suggestions to improve eye health. We also added an AI Coach that would suggest and demonstrate simple eye and neck exercises during short screen breaks to help reduce eye strain and neck fatigue. Finally, we reviewed the updated project file before preparing it for the final printout."),
("31st August 2026",
"After further development and discussion, we refined the project from the original AI OcuShield concept into <b>EyeGuard AI</b>, with a stronger focus on digital eye wellness and healthy screen habits. We focused on making the application a practical desktop tool that could use the webcam to monitor <b>blink rate and screen distance</b> and provide timely guidance. This helped us define a clearer and more focused direction for the project."),
("3rd September 2026",
"We continued the development of <b>EyeGuard AI</b> using <b>Python, OpenCV, and MediaPipe</b>. We worked on webcam-based face tracking, blink detection, and estimating the user's distance from the screen. We also continued improving the user interface and worked through technical compatibility and implementation issues encountered during development."),
("18th September 2026",
"We expanded the project by integrating an <b>AI Health Coach using Google Gemini</b>. The AI Coach was designed to use the user's eye-wellness information to provide more personalised suggestions. We also added the concept of <b>guided eye-relaxation exercises and desktop notifications</b>, making the application more interactive instead of only displaying measurements."),
("28th September 2026",
"We improved the AI coaching and wellness features of EyeGuard AI. The application was developed to work with data such as <b>blink rate, screen distance, session duration, and warnings</b>. We also worked on AI-generated recommendations, guided exercises, notifications, and statistics so that users could receive more useful and personalised feedback."),
("7th October 2026",
"We worked on making EyeGuard AI a more complete and polished desktop application. We planned and developed additional features including a <b>modern dashboard, graphs and historical statistics, adaptive AI coaching, personalised analysis, guided exercises, smart notifications, a Digital Eye Wellness Score, and system-tray support</b>. We also added an <b>Achievements tab</b> to recognise healthy user habits and make the application more engaging. We continued to focus on privacy by keeping webcam processing local and using only necessary numerical statistics for AI-based analysis.")
]

# Split into the same 3 original logbook pages plus continuation pages.
groups = [
    entries[0:3],
    entries[3:7],
    entries[7:12],
    entries[12:16],
    entries[16:18]
]

story = []

for page_index, group in enumerate(groups):
    story.append(Paragraph("LOG BOOK", title))

    rows = []
    for date, work in group:
        rows.append([
            Paragraph(f"<b>{date}</b>", date_style),
            Paragraph(work, body_style),
            Paragraph("", body_style)
        ])

    table = Table(rows, colWidths=[28*mm, 139*mm, 15*mm], repeatRows=0)
    table.setStyle(TableStyle([
        ("VALIGN", (0,0), (-1,-1), "TOP"),
        ("GRID", (0,0), (-1,-1), 0.6, colors.black),
        ("LEFTPADDING", (0,0), (-1,-1), 5),
        ("RIGHTPADDING", (0,0), (-1,-1), 5),
        ("TOPPADDING", (0,0), (-1,-1), 7),
        ("BOTTOMPADDING", (0,0), (-1,-1), 7),
    ]))
    story.append(table)

    if page_index < len(groups) - 1:
        story.append(PageBreak())

doc.build(story)
print(path)
