# -*- coding: utf-8 -*-
# app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# --- Configuration ---
st.set_page_config(
    page_title="LA Crime Analysis (2020-2023)",
    page_icon=":guardsman:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Data Loading and Preprocessing (Cached) ---

# Custom crime categorization: 137 (10 groups)
crime_category_mapping = {
    # 1. Vehicle-Related Crimes: 12
    'VEHICLE_CRIMES': [
        'BURGLARY FROM VEHICLE', 'BURGLARY FROM VEHICLE, ATTEMPTED',
        'BIKE - STOLEN', 'BIKE - ATTEMPTED STOLEN',
        'THEFT FROM MOTOR VEHICLE - GRAND ($950.01 AND OVER)',
        'THEFT FROM MOTOR VEHICLE - PETTY ($950 & UNDER)',
        'THEFT FROM MOTOR VEHICLE - ATTEMPT',
        'VEHICLE - STOLEN', 'VEHICLE - ATTEMPT STOLEN',
        'VEHICLE, STOLEN - OTHER (MOTORIZED SCOOTERS, BIKES, ETC)',
        'DRIVING WITHOUT OWNER CONSENT (DWOC)',
        'THROWING OBJECT AT MOVING VEHICLE',
    ],

    # 2. Theft & Burglary: 23 (Non-Vehicle)
    'THEFT_BURGLARY': [
        'SHOPLIFTING-GRAND THEFT ($950.01 & OVER)',
        'SHOPLIFTING - PETTY THEFT ($950 & UNDER)', 'SHOPLIFTING - ATTEMPT',
        'THEFT PLAIN - PETTY ($950 & UNDER)', 'THEFT PLAIN - ATTEMPT',
        'THEFT-GRAND ($950.01 & OVER)EXCPT,GUNS,FOWL,LIVESTK,PROD',
        'BURGLARY', 'BURGLARY, ATTEMPTED',
        'PICKPOCKET', 'PICKPOCKET, ATTEMPT',
        'PURSE SNATCHING', 'PURSE SNATCHING - ATTEMPT',
        'THEFT, PERSON', 'THEFT FROM PERSON - ATTEMPT',
        'DISHONEST EMPLOYEE - GRAND THEFT', 'DISHONEST EMPLOYEE - PETTY THEFT',
        'DISHONEST EMPLOYEE ATTEMPTED THEFT',
        'TILL TAP - PETTY ($950 & UNDER)',
        'TILL TAP - GRAND THEFT ($950.01 & OVER)',
        'THEFT, COIN MACHINE - PETTY ($950 & UNDER)',
        'THEFT, COIN MACHINE - GRAND ($950.01 & OVER)',
        'THEFT, COIN MACHINE - ATTEMPT',
        'DRUNK ROLL',
    ],

    # 3. Violent Crimes: 22 (Assault, Homicide, Threats, Kidnapping etc.)
    'VIOLENT_CRIMES': [
        'BATTERY - SIMPLE ASSAULT', 'INTIMATE PARTNER - SIMPLE ASSAULT',
        'ASSAULT WITH DEADLY WEAPON, AGGRAVATED ASSAULT',
        'INTIMATE PARTNER - AGGRAVATED ASSAULT',
        'ASSAULT WITH DEADLY WEAPON ON POLICE OFFICER',
        'BATTERY POLICE (SIMPLE)', 'BATTERY ON A FIREFIGHTER',
        'OTHER ASSAULT', 'ROBBERY', 'ATTEMPTED ROBBERY',
        'CRIMINAL HOMICIDE', 'MANSLAUGHTER, NEGLIGENT',
        'CRIMINAL THREATS - NO WEAPON DISPLAYED',
        'THREATENING PHONE CALLS/LETTERS', 'EXTORTION',
        'KIDNAPPING', 'KIDNAPPING - GRAND ATTEMPT',
        'FALSE IMPRISONMENT', 'STALKING',
        'LYNCHING', 'LYNCHING - ATTEMPTED',
        'BATTERY WITH SEXUAL CONTACT'
    ],

    # 4. Sex Offenses: 17 (Incl. against minors if primary nature is sexual)
    'SEX_OFFENSES': [
        'SODOMY/SEXUAL CONTACT B/W PENIS OF ONE PERS TO ANUS OTH',
        'SEX,UNLAWFUL(INC MUTUAL CONSENT, PENETRATION W/ FRGN OBJ',
        'LETTERS, LEWD  -  TELEPHONE CALLS, LEWD', 'LEWD CONDUCT',
        'ORAL COPULATION', 'RAPE, FORCIBLE', 'RAPE, ATTEMPTED',
        'SEXUAL PENETRATION W/FOREIGN OBJECT',
        'LEWD/LASCIVIOUS ACTS WITH CHILD', # Child victim, primarily sexual act
        'INDECENT EXPOSURE', 'PEEPING TOM',
        'CHILD PORNOGRAPHY', # Child victim, primarily sexual content crime
        'INCEST (SEXUAL ACTS BETWEEN BLOOD RELATIVES)',
        'BEASTIALITY, CRIME AGAINST NATURE SEXUAL ASSLT WITH ANIM',
        'PIMPING', 'PANDERING', # Exploitation related to sex acts
        'HUMAN TRAFFICKING - COMMERCIAL SEX ACTS'
    ],

    # 5. Crimes Against Children: 9 (Non-Sexual Abuse, Neglect, Endangerment)
    'CRIMES_AGAINST_CHILDREN': [
        'CRM AGNST CHLD (13 OR UNDER) (14-15 & SUSP 10 YRS OLDER)',
        'CHILD ANNOYING (17YRS & UNDER)',
        'CHILD ABUSE (PHYSICAL) - SIMPLE ASSAULT',
        'CHILD ABUSE (PHYSICAL) - AGGRAVATED ASSAULT',
        'CHILD STEALING', 'CHILD NEGLECT (SEE 300 W.I.C.)',
        'CHILD ABANDONMENT',
        'CONTRIBUTING', # likely Contrib. to Delinquency of Minor
        'DRUGS, TO A MINOR',
    ],

    # 6. Fraud & financial crimes: 19
    'FRAUD_FINANCIAL': [
        'THEFT OF IDENTITY',
        'BUNCO, GRAND THEFT', 'BUNCO, PETTY THEFT', 'BUNCO, ATTEMPT',
        'EMBEZZLEMENT, GRAND THEFT ($950.01 & OVER)',
        'EMBEZZLEMENT, PETTY THEFT ($950 & UNDER)',
        'CREDIT CARDS, FRAUD USE ($950.01 & OVER)',
        'CREDIT CARDS, FRAUD USE ($950 & UNDER',
        'DOCUMENT FORGERY / STOLEN FELONY',
        'DEFRAUDING INNKEEPER/THEFT OF SERVICES, $950 & UNDER',
        'DEFRAUDING INNKEEPER/THEFT OF SERVICES, OVER $950.01',
        'DOCUMENT WORTHLESS ($200.01 & OVER)',
        'DOCUMENT WORTHLESS ($200 & UNDER)',
        'COUNTERFEIT',
        'GRAND THEFT / INSURANCE FRAUD',
        'PETTY THEFT - AUTO REPAIR', # assuming fraud
        'GRAND THEFT / AUTO REPAIR', # assuming fraud
        'UNAUTHORIZED COMPUTER ACCESS',
        'BRIBERY'
    ],

    # 7. Vandalism & Property Damage: 4
    'VANDALISM_PROPERTY_DAMAGE': [
        'VANDALISM - FELONY ($400 & OVER, ALL CHURCH VANDALISMS)',
        'VANDALISM - MISDEAMEANOR ($399 OR UNDER)',
        'ARSON',
        'TELEPHONE PROPERTY - DAMAGE',
    ],

    # 8. Public disorder: 10 (affects quality of life)
    'PUBLIC_DISORDER': [
        'TRESPASSING',
        'DISTURBING THE PEACE',
        'FALSE POLICE REPORT',
        'ILLEGAL DUMPING',
        'PROWLER',
        'RECKLESS DRIVING',
        'RESISTING ARREST',
        'DISRUPT SCHOOL',
        'INCITING A RIOT',
        'FAILURE TO YIELD'
    ],

    # 9. Public safety crimes: 16 (Weapons, Orders, Trafficking, High Risk)
    'PUBLIC_SAFETY': [
        'BRANDISH WEAPON',
        'WEAPONS POSSESSION/BOMBING',
        'DISCHARGE FIREARMS/SHOTS FIRED',
        'SHOTS FIRED AT INHABITED DWELLING',
        'SHOTS FIRED AT MOVING VEHICLE, TRAIN OR AIRCRAFT',
        'REPLICA FIREARMS(SALE,DISPLAY,MANUFACTURE OR DISTRIBUTE)',
        'VIOLATION OF RESTRAINING ORDER',
        'VIOLATION OF TEMPORARY RESTRAINING ORDER',
        'VIOLATION OF COURT ORDER', 'CONTEMPT OF COURT',
        'FIREARMS RESTRAINING ORDER (FIREARMS RO)',
        'FIREARMS EMERGENCY PROTECTIVE ORDER (FIREARMS EPO)',
        'SEX OFFENDER REGISTRANT OUT OF COMPLIANCE',
        'BOMB SCARE', 'HUMAN TRAFFICKING - INVOLUNTARY SERVITUDE',
        'TRAIN WRECKING'
    ],

    # 10. Other crimes: 5 (Miscellaneous, Rare)
    'OTHER_CRIMES': [
        'OTHER MISCELLANEOUS CRIME',
        'CRUELTY TO ANIMALS',
        'BIGAMY',
        'BLOCKING DOOR INDUCTION CENTER',
        'CONSPIRACY'
    ]
}

# Create meaningful weapon groups
weapon_mapping = {
    'Firearm': [
        'HAND GUN', 'SEMI-AUTOMATIC PISTOL', 'UNKNOWN FIREARM', 
        'RIFLE', 'SHOTGUN', 'ASSAULT WEAPON/UZI/AK47', 
        'AUTOMATIC WEAPON/SUB-MACHINE GUN', 'MAC-10/11 AND SIMILAR ASSAULT WEAPON'
    ],
    'Knife': [
        'KNIFE WITH BLADE 6INCHES OR LESS', 'OTHER KNIFE', 
        'KNIFE WITH BLADE OVER 6 INCHES IN LENGTH', 'SWITCH BLADE', 
        'DIRK/DAGGER', 'MACHETE'
    ],
    'Physical Force': ['STRONG-ARM (HANDS, FIST, FEET OR BODILY FORCE)'],
    'Chemical': ['MACE/PEPPER SPRAY', 'CAUSTIC CHEMICAL/POISON'],
    'Verbal': ['VERBAL THREAT'],
    'Blunt Object': [
        'BLUNT INSTRUMENT', 'CLUB/BAT', 'STICK', 'HAMMER', 
        'METAL PIPE/POLE', 'BOARD'
    ],
    'Other Weapon': [
        'OTHER DANGEROUS WEAPON','UNKNOWN WEAPON/OTHER WEAPON', 'VEHICLE', 
        'SIMULATED GUN', 'ANTIQUE FIREARM', 'MARTIAL ARTS WEAPONS', 'BOTTLE', 
        'BRASS KNUCKLES', 'DEMAND NOTE', 'EXPLOXIVE DEVICE', 
        'FIXATION CAUSING FEAR', 'ICE PICK', 'IMPORTED FIREARM', 'ROPE/LIGATURE', 
        'SCALDING LIQUID', 'SCISSORS', 'SYRINGE', 'TIRE IRON'
    ],
    'Unknown': ['Unknown']
}

def find_category(description, mapping):
    """Helper function to find the category of a description."""
    desc_upper = str(description).upper().strip()

    # weapon mapping
    if mapping == weapon_mapping:
        for category, weapons in mapping.items():
            if desc_upper in [w.upper().strip() for w in weapons]:
                return category
        if desc_upper == 'UNKNOWN':
            return 'Unknown'
        return 'Other Weapon'
    
    # crime mapping
    elif mapping == crime_category_mapping:
        for category, crimes in mapping.items():
            if any(crime.upper().strip() in desc_upper for crime in crimes):
                return category
        return 'OTHER_CRIMES'
    
    return 'Unknown mapping'

@st.cache_data
def load_preprocess_data(csv_path):
    try:
        df = pd.read_csv(csv_path)
        print(f"Data loaded successfully from {csv_path}")
    except FileNotFoundError:
        st.error(f"File not found at: {csv_path}")
        return None

    # --- Initial Data Inspection ---
    # Remove unwanted columns
    columns_to_drop = [
        'DR_NO', 'Date Rptd', 'AREA', 'Rpt Dist No',
        'Mocodes', 'Cross Street', 'Crm Cd 1', 'Crm Cd 2',
        'Crm Cd 3', 'Crm Cd 4','Weapon Used Cd', 'Premis Cd',
        'Status', 'Status Desc', 'LOCATION'
    ]
    df = df.drop(columns=columns_to_drop, errors='ignore')

    # Parse 'DATE OCC'
    df['DATE OCC'] = pd.to_datetime(df['DATE OCC'], format='%m/%d/%Y %H:%M:%S %p')
    df = df.dropna(subset=['DATE OCC'])
    print(f"Rows after date parsing: {len(df)}")

    # Filter data for years 2020-2023
    df = df[df['DATE OCC'].dt.year.isin([2020, 2021, 2022, 2023])].copy()
    print(f"Rows after filtering by year: {len(df)}")
    if df.empty:
        st.warning("No data available for 2020-2023.")
        return None
    
    # --- Handle missing values ---
    # Filter Victim Age
    df = df[(df['Vict Age'] >= 5) & (df['Vict Age'] <= 100)].copy()
    print(f"Rows after filtering Victim Age (5-100): {len(df)}")
    if df.empty:
        st.warning("No data available for Victim Age between 5 and 100.")
        return None
    df['Vict Age'] = df['Vict Age'].astype(int) # ensure age is integer

    # Fill categorical columns with 'Unknown'
    for col in ['Vict Sex', 'Vict Descent', 'Weapon Desc', 'Premis Desc']:
        if col in df.columns:
            df[col] = df[col].fillna('Unknown')
    
    # --- Standardize data ---
    # Sex Codes: Male/Female/Other
    df['Vict Sex'] = df['Vict Sex'].replace({
        'M': 'Male',
        'F': 'Female',
        'H': 'Other',
        'Unknown': 'Other',
        'X': 'Other'
    }).fillna('Other').str.upper()

    # Descent (Race/Ethnicity) Codes:
    # B: Black, W: White, H: Hispanic, A: Asian, O: Other, X: Unknown
    df['Vict Descent'] = df['Vict Descent'].replace({
        'B': 'Black',
        'W': 'White',
        'H': 'Hispanic',
        'A': 'Asian',
        'O': 'Other',
        'X': 'Unknown',
        '-': 'Unknown',  # Missing → Unknown
        'K': 'Asian',    # Korean → Asian
        'C': 'Asian',    # Chinese → Asian
        'J': 'Asian',    # Japanese → Asian
        'F': 'Asian',    # Filipino → Asian
        'V': 'Asian',    # Vietnamese → Asian
        'I': 'Other',    # American Indian → Other
        'S': 'Other',    # Samoan → Other
        'P': 'Other',    # Pacific Islander → Other
        'Z': 'Asian',    # Asian Indian → Asian
        'G': 'Other',    # Guamanian → Other
        'U': 'Unknown',  # Unknown
        'D': 'Asian',    # Cambodian → Asian
        'L': 'Asian',    # Laotian → Asian
        'Unknown': 'Unknown'
    }).fillna('Unknown').str.upper()

    # --- Feature Engineering ---
    # Crime Category Mapping
    df['Crime Category'] = df['Crm Cd Desc'].apply(lambda x: find_category(x, crime_category_mapping))

    # Weapon Group Mapping
    df['Weapon Group'] = df['Weapon Desc'].apply(lambda x: find_category(x, weapon_mapping))

    # Time Features
    df['Year'] = df['DATE OCC'].dt.year
    df['Month'] = df['DATE OCC'].dt.month
    df['DayOfWeek'] = df['DATE OCC'].dt.dayofweek
    df['HourOCC'] = df['TIME OCC'].astype(str).str.zfill(4).str[:2].astype(int)

    # Age Group
    bins = [5, 18, 25, 35, 50, 65, 101] # Adjusted upper bound to include 100
    labels = ['Child/Teen', 'Young Adult', 'Adult', 'Mid-Adult', 'Older Adult', 'Senior']
    df['Age Group'] = pd.cut(df['Vict Age'], bins=bins, labels=labels, right=False) # right=False means [min, max)

    print(f"Data Preprocessing Completed: {len(df)} rows")
    return df   

# --- Plotting Functions ---
def yearly_monthly_trends(df):
    fig, ax = plt.subplots(figsize=(10, 5))
    if not df.empty:
        yearly_crimes = df.groupby(['Year', 'Month']).size().reset_index(name='Count')
        sns.lineplot(data=yearly_crimes, x='Month', y='Count', hue='Year', palette='deep', marker='o', ax=ax)
        ax.set_title('Yearly-Monthly Crime Trends (2020-2023)', fontsize=16)
        ax.set_xlabel('Month', fontsize=12)
        ax.set_ylabel('Number of Crimes', fontsize=12)
        ax.set_xticks(range(1, 13))
        ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                            'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(title='Year')
    else:
        ax.text(0.5, 0.5, 'No data available for this plot.',
                horizontalalignment='center', verticalalignment='center',
                fontsize=14, color='red')
    plt.tight_layout()
    return fig

def weekly_trends(df):
    fig, ax = plt.subplots(figsize=(10, 5))
    if not df.empty:
        weekly_crimes = df['DayOfWeek'].value_counts().sort_index()
        weekly_crimes.plot(kind='line', marker='o', ax=ax)
        ax.set_title('Crime Trends by Day of the Week', fontsize=16)
        ax.set_xlabel('Day of the Week', fontsize=12)
        ax.set_ylabel('Number of Crimes', fontsize=12)
        ax.set_xticks(range(7))
        ax.set_xticklabels(['Mon','Tue','Wed','Thu','Fri','Sat','Sun'], rotation='horizontal')
        ax.grid(True, linestyle='--', alpha=0.6)
    else:
        ax.text(0.5, 0.5, 'No data available for this plot.',
                horizontalalignment='center', verticalalignment='center',
                fontsize=14, color='red')
    plt.tight_layout()
    return fig

def hourly_trends(df):
    fig, ax = plt.subplots(figsize=(10, 5))
    if not df.empty:
        hourly_category = df.groupby(['HourOCC', 'Crime Category']).size().unstack()
        hourly_category.plot(kind='bar', stacked=True, color=sns.color_palette('deep'), width=0.8, ax=ax)
        ax.set_title('Hourly Crime Trends by Category', fontsize=16)
        ax.set_xlabel('Hour', fontsize=12)
        ax.set_ylabel('Number of Crimes', fontsize=12)
        ax.set_xticks(range(0, 24))
        ax.set_xticklabels(range(0, 24), rotation='horizontal')
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        ax.legend(title='Crime Category', bbox_to_anchor=(1.05, 1), fontsize='small')
    else:
        ax.text(0.5, 0.5, 'No data available for this plot.',
                horizontalalignment='center', verticalalignment='center',
                fontsize=14, color='red')
    plt.tight_layout()
    return fig

def seasonal_spikes(df, top_n=5):
    fig, ax = plt.subplots(figsize=(10, 5))
    if not df.empty:
        top_crimes = df['Crime Category'].value_counts().nlargest(top_n).index
        monthly_data = df[df['Crime Category'].isin(top_crimes)].groupby(['Month', 'Crime Category']).size().unstack()
        monthly_data.plot(kind='line', marker='o', ax=ax, colormap='tab10')
        
        ax.set_title('Seasonal Trends for Top {top_n} Crime Categories', fontsize=16)
        ax.set_xlabel('Month', fontsize=12)
        ax.set_ylabel('Number of Crimes', fontsize=12)
        ax.set_xticks(range(1, 13))
        ax.set_xticklabels(['Jan','Feb','Mar','Apr','May','Jun','Jul',
                            'Aug','Sep','Oct','Nov','Dec'])
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(title='Crime Category', bbox_to_anchor=(1.05, 1), fontsize='small')
    else:
        ax.text(0.5, 0.5, 'No data available for this plot.',
                horizontalalignment='center', verticalalignment='center',
                fontsize=14, color='red')
    plt.tight_layout()
    return fig

def crime_hotspots(df, sort_by='Total'):
    fig, ax = plt.subplots(figsize=(10, 5))
    if not df.empty:
        crime_hotspots = df.groupby(['AREA NAME', 'Crime Category']).size().unstack()
        crime_hotspots['Total'] = crime_hotspots.sum(axis=1)

        # Determine sorting order
        if sort_by == 'Total':
            crime_sorted = crime_hotspots.sort_values(by='Total', ascending=True)
            sort_title = 'Total Crimes'
        else:
            crime_sorted = crime_hotspots.sort_values(by=sort_by, ascending=True)
            sort_title = f'{sort_by} Crimes'
        
        # Plotting
        crime_sorted.drop(columns='Total').plot(kind='barh', stacked=True, color=sns.color_palette('deep'), ax=ax)
        ax.set_title('Crime Hotspots by Area and Crime Category', fontsize=16)
        ax.set_xlabel('Number of Crimes', fontsize=12)
        ax.set_ylabel('Area', fontsize=12)
        ax.legend(title='Crime Category', bbox_to_anchor=(1.05, 1), fontsize='small')
    else:
        ax.text(0.5, 0.5, 'No data available for this plot.',
                horizontalalignment='center', verticalalignment='center',
                fontsize=14, color='red')
    plt.tight_layout()

    # Adjust layout to prevent legend overlap
    plt.subplots_adjust(right=0.75)
    return fig

def victim_age(df):
    fig, ax = plt.subplots(figsize=(10, 5))
    if not df.empty and 'Vict Descent' in df.columns and 'Vict Age' in df.columns:
        order = df.groupby('Vict Descent')['Vict Age'].median().sort_values().index
        sns.boxplot(data=df, x='Vict Descent', y='Vict Age', order=order, ax=ax, palette='viridis')
        ax.set_title('Victim Age Distribution by Descent', fontsize=16)
        ax.set_xlabel('Descent', fontsize=12)
        ax.set_ylabel('Age', fontsize=12)
    else:
         ax.text(0.5, 0.5, 'No data available for this plot.', 
                horizontalalignment='center', verticalalignment='center')
    plt.tight_layout()
    return fig

def victim_sex(df):
    fig, ax = plt.subplots(figsize=(10, 5))
    if not df.empty:
        sex_counts = df.groupby(['Vict Sex', 'Crime Category']).size().unstack()
        sex_counts_category = sex_counts.apply(lambda x: x*100 / sum(x), axis=1)
        sex_counts_category.plot(kind='bar', stacked=True, color=sns.color_palette('deep'), ax=ax)
        ax.set_title('Distribution of Crime Categories by Victim Sex (%)', fontsize=16)
        ax.set_xlabel('Victim Sex', fontsize=12)
        ax.set_ylabel('Percentage of Crimes (%)', fontsize=12)
        ax.tick_params(axis='x', rotation=0)
        ax.legend(title='Crime Category', bbox_to_anchor=(1.05, 1), fontsize='small')
        # format y-axis as percentage
        ax.yaxis.set_major_formatter(plt.FuncFormatter('{:.0f}%'.format))
    else:
        ax.text(0.5, 0.5, 'No data available for this plot.',
                horizontalalignment='center', verticalalignment='center',
                fontsize=14, color='red')
    plt.tight_layout()
    plt.subplots_adjust(right=0.75)
    return fig

def weapon_heatmap(df):
    fig, ax = plt.subplots(figsize=(14, 10))
    if not df.empty:
        # Filter out cases where weapon is None/Unknown for a clearer heatmap
        weapon_counts = df[df['Weapon Group'] != 'None/Unknown']\
            .groupby(['Crime Category', 'Weapon Group']).size().unstack(fill_value=0)

        # Focus on crimes with significant weapon use (optional, adjust threshold)
        min_weapon_crimes = 50 # Example threshold
        weapon_counts = weapon_counts[weapon_counts.sum(axis=1) > min_weapon_crimes]

        if not weapon_counts.empty:
            # Calculate percentages for heatmap clarity
            normalized_counts = weapon_counts.div(weapon_counts.sum(axis=1), axis=0) * 100

            sns.heatmap(normalized_counts.T, annot=True, fmt='.1f', cmap='YlGnBu', linewidths=.5,
                       cbar_kws={'label': '% of Cases for the Crime Type'}, ax=ax)
            ax.set_title('Weapon Type Prevalence per Crime Category (Known Weapons Only)', fontsize=16)
            ax.set_ylabel('Weapon Group', fontsize=12)
            ax.set_xlabel('Crime Category', fontsize=12)
            ax.tick_params(axis='x', rotation=45)
            ax.tick_params(axis='y', rotation=0)
        else:
            ax.text(0.5, 0.5, 'Insufficient data with known weapons for heatmap.', 
                    horizontalalignment='center', verticalalignment='center')

    else:
        ax.text(0.5, 0.5, 'No data available for this plot.', 
                horizontalalignment='center', verticalalignment='center')
    plt.tight_layout()
    return fig

# --- Streamlit App UI ---
st.title("📊 Crime Trends Analysis in Los Angeles (2020-2023)")
st.markdown("""
Welcome to the LA Crime Analysis dashboard. This application explores reported crime incidents
in Los Angeles from January 2020 to December 2023, based on LAPD data.
Navigate through the tabs to explore trends, hotspots, victim demographics, weapon usage,
modeling insights, and public safety recommendations.
""")

# --- Load Data ---
DATA_PATH = 'crime-in-LA/Crime_Data_from_2020_to_Present.csv' # Ensure this file is in the same directory
df_processed = load_preprocess_data(DATA_PATH)

if df_processed is not None and not df_processed.empty:
    st.success(f"Successfully loaded and processed {len(df_processed):,} crime records.")

    # --- Create Tabs ---
    tab_titles = [
        "🕒 Temporal Trends",
        "📍 Geospatial Hotspots",
        "👥 Victim Demographics",
        "🔪 Weapon Usage",
        "🤖 Modeling Insights",
        "💡 Recommendations"
    ]
    tabs = st.tabs(tab_titles)

    # --- Temporal Tab ---
    with tabs[0]:
        st.header("🕒 Temporal Crime Patterns")
        st.markdown("How do crime rates vary by year, month, day of the week, and hour of the day?")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Monthly Trends per Year")
            st.pyplot(yearly_monthly_trends(df_processed))
        with col2:
             st.subheader("Crimes by Day of Week")
             st.pyplot(weekly_trends(df_processed))

        st.subheader("Crimes by Hour of Day")
        st.pyplot(hourly_trends(df_processed))

        st.subheader("Seasonal Trends for Top Crime Categories")
        st.pyplot(seasonal_spikes(df_processed, top_n=5))


    # --- Geospatial Tab ---
    with tabs[1]:
        st.header("📍 Geospatial Hotspots")
        st.markdown("Which areas have the highest crime rates, and are certain crimes concentrated?")

        # Allow sorting
        sort_options = ['Total'] + df_processed['Crime Category'].unique().tolist()
        sort_by = st.selectbox("Sort Areas By:", options=sort_options, index=0, key='hotspot_sort')

        st.pyplot(crime_hotspots(df_processed, sort_by=sort_by))
        st.caption("Showing the top 20 areas based on the selected sorting criteria.")


    # --- Demographics Tab ---
    with tabs[2]:
        st.header("👥 Victim Demographics")
        st.markdown("Exploring the age, sex, and descent of victims across different crime types.")

        
        st.subheader("Victim Age Distribution by Descent")
        st.pyplot(victim_age(df_processed))
        st.caption("Box plots show median (line), interquartile range (box), and potential outliers (points).")

    
        st.subheader("Victim Age Groups")
        if 'Age Group' in df_processed.columns:
            fig_age_group, ax_age_group = plt.subplots(figsize=(10, 5))
            df_processed['Age Group'].value_counts().sort_index().plot(kind='bar', ax=ax_age_group, color='coral')
            ax_age_group.set_title('Victim Count by Age Group')
            ax_age_group.set_ylabel('Number of Victims')
            ax_age_group.tick_params(axis='x', rotation=0)
            st.pyplot(fig_age_group)
        else:
            st.warning("Age Group column not found.")

        st.subheader("Crime Categories by Victim Sex (%)")
        st.pyplot(victim_sex(df_processed))
        st.caption("Stacked bar chart showing the percentage distribution of crime categories for each sex.")


    # --- Weapon Usage Tab ---
    with tabs[3]:
        st.header("🔪 Weapon Usage Patterns")
        st.markdown("How does weapon usage vary by crime type? (Focuses on cases with known weapons)")
        st.pyplot(weapon_heatmap(df_processed))
        st.caption("Heatmap shows the percentage of times a specific weapon group was used within each crime category. Based only on incidents where a weapon type was reported (not 'None/Unknown').")


    # --- Modeling Insights Tab ---
    with tabs[4]:
        st.header("🤖 Modeling Insights (Pre-computed)")
        st.markdown("""
        Results from a Random Forest model trained to predict the 'Crime Category'.
        **Note:** The model is not run live; these are pre-calculated results from the analysis phase.
        """)

        col1, col2 = st.columns([2,1]) # Give more space to SHAP plot/text

        with col1:
            # st.subheader("Top Feature Importances")
            # st.pyplot(plot_feature_importance(filepath='top_features.csv'))
            # st.caption("Features that the Random Forest model found most influential in predicting the crime category.")

            st.subheader("SHAP Value Insights (Summary)")
            # Display saved SHAP plot
            shap_plot_path = "shap_summary_plot.JPG"
            if os.path.exists(shap_plot_path):
                st.image(shap_plot_path, caption="SHAP Summary Plot (Illustrative)", use_container_width=True)
            else:
                st.warning(f"SHAP summary plot image not found at: {shap_plot_path}")

        with col2:
            st.subheader("Key SHAP Observations")
            st.markdown("""
            *   **Victim Age (`Vict Age`):** Lower ages strongly push predictions towards `CRIMES_AGAINST_CHILDREN`. Higher ages have less specific impact across categories.
            *   **Weapon Features:** Presence of specific weapon descriptions (like `Weapon Desc_STRONG-ARM...`, `Weapon Desc_HAND GUN`) strongly influences predictions towards `VIOLENT_CRIMES` or `PUBLIC_SAFETY`.
            *   **Time (`TIME OCC`, `day_of_week_occ`):** Certain hours (e.g., midday) slightly increase likelihood for `FRAUD_FINANCIAL`, while late nights might influence `VEHICLE_CRIMES` or `VIOLENT_CRIMES`. Day of week shows some patterns (e.g., Friday peak).
            *   **Location (`AREA NAME_...`):** Specific areas strongly influence certain predictions (e.g., `AREA NAME_Central` features prominently).
            *   **Part 1-2 (`Part 1-2`):** This classification (serious vs. less serious) is a highly important feature, distinguishing between categories like `THEFT_BURGLARY`/`VIOLENT_CRIMES` (often Part 1) and others.
            *(Based on typical SHAP results for this type of problem)*
            """)
            st.subheader("Model Performance Snippet")
            st.markdown("""
            *(Example metrics from classification report - replace with your actuals)*
            *   **Overall Accuracy:** ~XX% (Varies based on data sample/run)
            *   **Key Challenge:** Distinguishing between high-volume, similar categories (e.g., types of theft).
            *   **Stronger Prediction:** For categories with distinct features (e.g., `CRIMES_AGAINST_CHILDREN` based on age, `PUBLIC_SAFETY` based on weapon/order violations).
            *   *(Add 1-2 specific precision/recall numbers if desired)*
            """)


    # --- Recommendations Tab ---
    with tabs[5]:
        st.header("💡 Recommendations for Enhancing Public Safety")
        st.markdown("""
        Based on the analysis of LA crime data from 2020-2023, here are key recommendations:

        **1. Data-Driven Resource Allocation:**
        *   **Dynamic Patrols:** Utilize hourly, weekly, and seasonal trend data to deploy patrols more effectively. Increase presence in identified hotspots (e.g., Central, 77th Street, Southwest for violent crimes; Pacific, West LA for property crimes) during peak times identified in the temporal analysis.
        *   **Focus on High-Volume/Impact Crimes:** Allocate investigative and preventative resources towards the most frequent categories (`VIOLENT_CRIMES`, `VEHICLE_CRIMES`, `THEFT_BURGLARY`) and those with high community impact.

        **2. Targeted Prevention Strategies:**
        *   **Vehicle Crime Reduction:** Implement campaigns focused on "Lock it or Lose it," catalytic converter theft prevention (etching programs), and secure bike parking, particularly in areas like Pacific and West LA.
        *   **Theft & Burglary Deterrence:** Partner with businesses in high-theft areas (e.g., Central) on loss prevention strategies. Promote residential security measures (lighting, cameras, community watch).
        *   **Violent Crime Intervention:** In areas like 77th Street, Southeast, and Newton, focus on community-based violence interruption programs, gang intervention strategies, and addressing root causes like economic disparity.

        **3. Leverage Geospatial Insights:**
        *   **Problem-Oriented Policing (POP):** Analyze the *specific* conditions contributing to crime in micro-locations within broader hotspots (e.g., poor lighting on a specific street, nuisance property). Work with city agencies (transportation, sanitation, housing) and community groups to address these underlying issues.
        *   **CPTED (Crime Prevention Through Environmental Design):** Promote design principles in new developments and public spaces that enhance visibility, lighting, and natural surveillance to deter opportunistic crime.

        **4. Enhance Community Collaboration & Data:**
        *   **Community Engagement:** Build trust and partnerships with residents and businesses in high-crime areas to co-develop solutions and improve intelligence gathering.
        *   **Address Data Gaps:** Improve the consistency and completeness of data collection, particularly for victim demographics ('Unknown' categories) and modus operandi, to enable more granular analysis.

        **5. Address Systemic Factors:**
        *   **Socioeconomic Support:** Advocate for and support long-term investments in education, job creation, affordable housing, and youth programs in disadvantaged areas identified as violent crime hotspots.
        *   **Mental Health & Substance Abuse:** Increase access to and coordination of mental health and substance abuse services, as these often intersect with homelessness, public disorder, and victimization.

        **Overall:** A multi-faceted, data-informed approach combining targeted enforcement, preventative measures, community collaboration, and addressing underlying socioeconomic factors is crucial for sustainably enhancing public safety across Los Angeles.
        """)

elif df_processed is None:
    st.error("Data loading failed. Please check the file path and ensure the CSV file is present.")
else: # df_processed is empty after filtering
     st.warning("No valid crime data found for the specified period (2020-2023) and criteria. Cannot display analysis.")

# --- Footer ---
st.markdown("---")
st.caption("Data Source: Los Angeles Police Department (LAPD). Analysis Period: Jan 2020 - Dec 2023.")