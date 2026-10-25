"""
generate_handbook.py
Generates a realistic, comprehensive, multi-page College Student Handbook PDF
for testing and demonstrating the LangChain AI College FAQ Chatbot.
"""

import os
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

def create_sample_handbook(output_path="data/college_handbook_sample.pdf"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        rightMargin=40, leftMargin=40,
        topMargin=40, bottomMargin=40
    )
    
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontSize=22,
        leading=26,
        textColor=colors.HexColor('#1E3A8A'),
        alignment=1, # Center
        spaceAfter=10
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontSize=12,
        leading=16,
        textColor=colors.HexColor('#4B5563'),
        alignment=1,
        spaceAfter=15
    )
    
    h1_style = ParagraphStyle(
        'SectionH1',
        parent=styles['Heading2'],
        fontSize=14,
        leading=18,
        textColor=colors.HexColor('#1E3A8A'),
        spaceBefore=14,
        spaceAfter=6,
        keepWithNext=True
    )
    
    h2_style = ParagraphStyle(
        'SectionH2',
        parent=styles['Heading3'],
        fontSize=11,
        leading=15,
        textColor=colors.HexColor('#1F2937'),
        spaceBefore=8,
        spaceAfter=4,
        keepWithNext=True
    )
    
    body_style = ParagraphStyle(
        'BodyDark',
        parent=styles['Normal'],
        fontSize=9.5,
        leading=13.5,
        textColor=colors.HexColor('#374151'),
        spaceAfter=6
    )
    
    bullet_style = ParagraphStyle(
        'BulletText',
        parent=body_style,
        leftIndent=15,
        firstLineIndent=-10,
        spaceAfter=4
    )

    story = []
    
    # Title Header
    story.append(Paragraph("APEX INSTITUTE OF TECHNOLOGY & SCIENCE", title_style))
    story.append(Paragraph("OFFICIAL STUDENT ACADEMIC HANDBOOK & FAQ GUIDE (ACADEMIC YEAR 2025–2026)", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor('#1E3A8A'), spaceAfter=15))
    
    # SECTION 1
    story.append(Paragraph("SECTION 1: ADMISSIONS & ELIGIBILITY", h1_style))
    story.append(Paragraph("<b>1.1 Undergraduate Admission Criteria:</b> Candidates seeking admission to B.Tech/B.E. programs must have passed 10+2 (Higher Secondary) with Physics, Mathematics, and Chemistry/Computer Science with a minimum aggregate of 60% (55% for reserved SC/ST/OBC categories). Valid scores in national entrance exams (JEE Main) or State CET are mandatory for merit allocation.", body_style))
    story.append(Paragraph("<b>1.2 Mandatory Document Checklist for Verification:</b>", h2_style))
    story.append(Paragraph("• Original and 3 attested copies of 10th and 12th Marksheets & Passing Certificates.", bullet_style))
    story.append(Paragraph("• Valid Entrance Examination Rank Card & Seat Allotment Letter.", bullet_style))
    story.append(Paragraph("• Transfer Certificate (TC) and Migration Certificate from the previous institution.", bullet_style))
    story.append(Paragraph("• Category / Caste Certificate (if applicable, issued by competent state authority).", bullet_style))
    story.append(Paragraph("• Aadhaar Card copy and 6 recent passport-sized photographs.", bullet_style))
    story.append(Paragraph("<b>1.3 Admission Cancellation & Fee Refund Policy:</b>", h2_style))
    story.append(Paragraph("• Withdrawal 15 days or more before the formal commencement of classes: 100% refund of tuition fee minus a processing charge of ₹1,000.", bullet_style))
    story.append(Paragraph("• Withdrawal within 15 days after commencement of classes: 80% tuition fee refund.", bullet_style))
    story.append(Paragraph("• Withdrawal after 30 days from commencement of classes: No tuition fee refund is admissible under any circumstances.", bullet_style))
    
    # SECTION 2
    story.append(Spacer(1, 10))
    story.append(Paragraph("SECTION 2: TUITION FEES & FINANCIAL POLICIES", h1_style))
    story.append(Paragraph("<b>2.1 Annual Fee Schedule:</b>", h2_style))
    
    fee_data = [
        ['Program', 'Tuition Fee (Per Sem)', 'Development Fee (Annual)', 'Library & Lab Deposit (Refundable)'],
        ['B.Tech (CSE, AI, IT)', '₹65,000', '₹12,000', '₹5,000 (One-time)'],
        ['B.Tech (ECE, Mech, Civil)', '₹55,000', '₹10,000', '₹5,000 (One-time)'],
        ['M.Tech / MCA', '₹45,000', '₹8,000', '₹5,000 (One-time)']
    ]
    t = Table(fee_data, colWidths=[150, 120, 130, 130])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1E3A8A')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,0), 9),
        ('BOTTOMPADDING', (0,0), (-1,0), 6),
        ('BACKGROUND', (0,1), (-1,-1), colors.HexColor('#F3F4F6')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#D1D5DB')),
        ('FONTNAME', (0,1), (-1,-1), 'Helvetica'),
        ('FONTSIZE', (0,1), (-1,-1), 8.5),
        ('ALIGN', (1,0), (-1,-1), 'CENTER'),
    ]))
    story.append(t)
    story.append(Spacer(1, 6))
    story.append(Paragraph("<b>2.2 Payment Deadlines & Late Fine Regulations:</b> Tuition fees for Odd Semesters must be paid by July 31st; Even Semesters must be paid by January 15th. A grace period of 7 days is permitted. Beyond the grace period, a late fine of ₹100 per calendar day is levied. Students with pending dues exceeding 30 days will be debarred from attending lab sessions and end-semester examinations.", body_style))
    story.append(Paragraph("<b>2.3 Scholarships & Concessions:</b> Merit scholarships (25% fee waiver) are awarded to students maintaining a CGPA of 9.0 or higher. Need-based financial assistance (Merit-cum-Means) is available for students whose family annual income is below ₹3,00,000, subject to verification by the Dean of Student Welfare.", body_style))
    
    # SECTION 3
    story.append(Spacer(1, 10))
    story.append(Paragraph("SECTION 3: ATTENDANCE MANDATE & LEAVE RULES", h1_style))
    story.append(Paragraph("<b>3.1 Minimum Attendance Requirement:</b> Every registered student must maintain a minimum of 75% attendance in aggregate across all lecture, tutorial, and laboratory hours to be eligible to sit for the End-Semester Examination.", body_style))
    story.append(Paragraph("<b>3.2 Attendance Condonation (Medical Exemption):</b>", h2_style))
    story.append(Paragraph("• Condonation is strictly limited to students having attendance between 65% and 74.9%.", bullet_style))
    story.append(Paragraph("• The condonation request must be submitted with valid government hospital medical certificates within 3 working days of returning to campus.", bullet_style))
    story.append(Paragraph("• A mandatory condonation processing fee of ₹500 per subject applies upon approval by the Academic Council.", bullet_style))
    story.append(Paragraph("• Students with attendance below 65% are NOT eligible for condonation under any circumstances and must repeat the semester courses as backlogs.", bullet_style))
    story.append(Paragraph("<b>3.3 Duty Leave (OD - On Duty):</b> Students representing the college in official hackathons, sports tournaments, conferences, or cultural fests are granted On-Duty (OD) attendance credit up to a maximum of 10 working days per academic semester with prior written approval from their Head of Department (HOD).", body_style))

    # SECTION 4
    story.append(Spacer(1, 10))
    story.append(Paragraph("SECTION 4: EXAMINATION SCHEME & GRADING RULES", h1_style))
    story.append(Paragraph("<b>4.1 Evaluation Weightage:</b> Total subject assessment comprises Continuous Internal Assessment (CIA - 40 Marks) and End-Semester University Examination (ESE - 60 Marks). Internal evaluation includes Mid-Term Exams (20 marks), Quizzes/Assignments (10 marks), and Class Participation & Attendance (10 marks).", body_style))
    story.append(Paragraph("<b>4.2 Minimum Passing Threshold:</b> A student must secure at least 40% in CIA and 40% in ESE independently, with an overall aggregate of 45% to successfully pass a course.", body_style))
    story.append(Paragraph("<b>4.3 Supplementary & Backlog Exams:</b> Supplementary examinations are conducted once every academic year during the summer recess (June-July). A student may register for a maximum of 4 backlog subjects per supplementary session by paying a fee of ₹750 per subject.", body_style))
    story.append(Paragraph("<b>4.4 Re-evaluation & Script Inspection:</b> Students unsatisfied with their semester grade may apply for re-evaluation within 15 calendar days of result announcement. Fees: ₹300 for Xerox copy of evaluated answer booklet, and ₹600 for complete re-checking by an external examiner.", body_style))
    story.append(Paragraph("<b>4.5 Academic Malpractice / Unfair Means:</b> Possession of mobile phones, smartwatches, cheat sheets, or impersonation during examinations results in immediate confiscation, debarment from the remaining exams of the semester, and referral to the Examination Disciplinary Committee.", body_style))

    # SECTION 5
    story.append(Spacer(1, 10))
    story.append(Paragraph("SECTION 5: CAMPUS CONDUCT & HOSTEL REGULATIONS", h1_style))
    story.append(Paragraph("<b>5.1 Anti-Ragging Policy:</b> Ragging in any form (physical, verbal, or mental harassment) is strictly prohibited. The institution enforces zero-tolerance under UGC Regulations 2009 and IPC. Any student found guilty faces immediate suspension, expulsion from the hostel, FIR with local police, and permanent debarment from all Indian universities.", body_style))
    story.append(Paragraph("<b>5.2 Hostel Timings & Night Out Passes:</b>", h2_style))
    story.append(Paragraph("• Hostel curfew time is strictly 9:00 PM on weekdays and 9:30 PM on weekends.", bullet_style))
    story.append(Paragraph("• Leaving campus overnight requires an electronic Night-Out Pass approved by the Chief Warden and verified via SMS/Call confirmation from registered parents.", bullet_style))
    story.append(Paragraph("• Unauthorized absence after curfew attracts a fine of ₹1,000 for the first offense and expulsion from hostel for repeated violations.", bullet_style))
    story.append(Paragraph("<b>5.3 Dress Code & Campus Identity:</b> Students must wear valid RFID College Identity Cards visibly around their neck while inside campus premises. Casual sandals, flip-flops, and offensive graphic t-shirts are prohibited in academic blocks and laboratories.", body_style))
    story.append(Paragraph("<b>5.4 Library Facilities & Book Loan Rules:</b> Students may borrow up to 4 books for a period of 14 days. An overdue fine of ₹5 per day per book applies thereafter. Reference books, rare editions, and current journals cannot be checked out.", body_style))

    # SECTION 6
    story.append(Spacer(1, 10))
    story.append(Paragraph("SECTION 6: KEY ADMINISTRATIVE CONTACTS & OFFICE HOURS", h1_style))
    contacts_data = [
        ['Office / Authority', 'Location', 'Email Address', 'Office Hours'],
        ['Office of Academic Affairs & Admissions', 'Admin Block, Ground Floor', 'admissions@apex-tech.edu', 'Mon-Fri: 9:00 AM - 4:30 PM'],
        ['Controller of Examinations (COE)', 'Exam Block, Room 204', 'coe@apex-tech.edu', 'Mon-Fri: 10:00 AM - 3:00 PM'],
        ['Accounts & Student Fee Section', 'Admin Block, Room 102', 'fees@apex-tech.edu', 'Mon-Fri: 9:30 AM - 2:00 PM'],
        ['Chief Hostel Warden', 'Hostel Block A, Ground Floor', 'warden@apex-tech.edu', 'Daily: 5:00 PM - 8:30 PM'],
        ['Student Grievance & Anti-Ragging Cell', 'Main Block, Room 310', 'antiragging@apex-tech.edu', '24x7 Helpline: 1800-425-9999']
    ]
    t2 = Table(contacts_data, colWidths=[140, 110, 140, 140])
    t2.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1E3A8A')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,0), 8.5),
        ('BOTTOMPADDING', (0,0), (-1,0), 5),
        ('BACKGROUND', (0,1), (-1,-1), colors.HexColor('#F9FAFB')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#E5E7EB')),
        ('FONTNAME', (0,1), (-1,-1), 'Helvetica'),
        ('FONTSIZE', (0,1), (-1,-1), 8),
    ]))
    story.append(t2)
    
    doc.build(story)
    print(f"Sample handbook created successfully at: {output_path}")

if __name__ == "__main__":
    create_sample_handbook()
