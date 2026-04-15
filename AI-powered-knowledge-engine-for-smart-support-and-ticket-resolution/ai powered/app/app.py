from flask import Flask, render_template, request, redirect, session, url_for, flash
import ticket_service, database, os

app = Flask(__name__)
app.secret_key = "ai_resolution_secret_key"

# Initialize the database and tables
database.init_db()

# --- JINJA FILTERS (Connecting Python logic to HTML) ---

@app.template_filter('confidence_info')
def confidence_info(score):
    """
    Matches the name used in my_tickets.html.
    Returns the label (text) and the CSS class (color).
    """
    label, css_class = ticket_service.confidence_label(score or 0.0)
    return {'label': label, 'class': css_class}

@app.template_filter('clean_md')
def clean_md(text):
    """Uses the mentor's logic to clean up AI resolution text indents"""
    return ticket_service.normalize_markdown(text)

# --- ROUTES ---

@app.route('/')
@app.route('/index')
def index():
    """Home/Login Page"""
    if 'user' in session:
        return redirect(url_for('dashboard'))
    return render_template('index.html')

@app.route('/login', methods=['POST'])
def login():
    """Handles authentication for the demo"""
    username = request.form.get('username')
    password = request.form.get('password')
    
    if username == 'admin' and password == 'admin123':
        session['user'] = 'admin'
        return redirect(url_for('dashboard'))
    
    flash("Invalid Credentials")
    return redirect(url_for('index'))

@app.route('/dashboard')
def dashboard():
    """Admin view with Chart.js and Knowledge Gap Heatmap"""
    if 'user' not in session:
        return redirect(url_for('index'))
    
    kpis = ticket_service.get_admin_kpis()
    labels, values, gaps = ticket_service.get_analytics_data()
    
    return render_template('dashboard.html', 
                           user=session['user'], 
                           kpis=kpis, 
                           labels=labels, 
                           values=values, 
                           gaps=gaps)

@app.route('/create_ticket', methods=['GET', 'POST'])
def create_ticket():
    """Main RAG interface for submitting tickets"""
    if 'user' not in session: 
        return redirect(url_for('index'))
        
    if request.method == 'POST':
        ticket = ticket_service.submit_ticket(
            request.form.get('title'), 
            request.form.get('description'), 
            request.form.get('category'), 
            request.form.get('priority'), 
            session['user']
        )
        return render_template('ai_solution.html', ticket=ticket)
        
    return render_template('create_ticket.html')

@app.route('/my_tickets')
def my_tickets():
    """History page using the Mentor's Ticket Card style"""
    if 'user' not in session:
        return redirect(url_for('index'))
    
    tickets_df = ticket_service.get_user_tickets(session['user'])
    # Convert dataframe to list of dicts for the HTML loop
    tickets = tickets_df.to_dict('records')
    
    return render_template('my_tickets.html', tickets=tickets)

@app.route('/logout')
def logout():
    """Logs out the user and clears session data"""
    session.clear()
    return redirect(url_for('index'))

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)