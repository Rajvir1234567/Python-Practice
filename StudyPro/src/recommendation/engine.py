def generate_recommendations(risk_classification, input_dict):
    """
    Generates prioritized, actionable learning interventions based on 
    risk classification and specific student feature thresholds.
    """
    recommendations = []
    
    studytime = input_dict.get('studytime', 2)
    failures = input_dict.get('failures', 0)
    absences = input_dict.get('absences', 0)
    g1 = input_dict.get('G1', 10)
    g2 = input_dict.get('G2', 10)
    
    if risk_classification == "Safe":
        recommendations.append("✅ **Academic Maintenance**: Maintain current study schedule and class engagement.")
        if studytime >= 3:
            recommendations.append("⭐ **Peer Mentorship**: Encourage student to participate as a peer tutor in weak subjects.")
        return recommendations

    # Interventions for Moderate & High Risk
    if absences > 10:
        recommendations.append(f"⚠️ **Attendance Warning**: Student has {absences} absences. Schedule an immediate academic counseling session to establish mandatory class attendance goals.")
        
    if failures > 0:
        recommendations.append(f"📚 **Remedial Coursework**: Student has {failures} prior course failure(s). Enroll in structured weekend refresher modules for foundational topics.")
        
    if studytime < 2:
        recommendations.append("⏱️ **Study Habit Enhancement**: Weekly study hours are low (< 5 hrs/week). Implement a structured 45-minute daily study block with academic planner tracking.")
        
    if g2 < 10 or g1 < 10:
        recommendations.append(f"📝 **Internal Assessment Support**: Term marks (G1: {g1}, G2: {g2}) are below passing threshold. Provide targeted mock test papers and focused instructor feedback.")
        
    if not recommendations:
        recommendations.append("🔍 **General Intervention**: Recommend bi-weekly progress reviews with subject teachers to monitor performance trajectory.")
        
    return recommendations
