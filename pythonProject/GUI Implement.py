import re
import numpy as np
import pandas as pd
import time
pd.set_option('display.max_columns',15)
from pathlib import Path
from tkinter import Tk, Canvas, Entry, Text, Button, PhotoImage, ttk, filedialog, messagebox
from scipy.stats import entropy
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from tabulate import tabulate
from datetime import datetime
from IPython.display import display


OUTPUT_PATH = Path(__file__).parent
ASSETS_PATH = OUTPUT_PATH / Path(r"/Users/antoniusbagus/Library/CloudStorage/OneDrive-UniversitasSanataDharma/Skripsi/Codingan Skripsi/pythonProject/assets/frame0")

def information_gain(x, y):
    igResult = []
    columnName = x.columns
    count = 0
    for element in columnName:
        columnValue = x[columnName[count]]
        entropy_before = entropy(columnValue.value_counts(normalize=True))
        y.name = 'split'
        columnValue.name = 'members'
        grouped_distrib = columnValue.groupby(y).value_counts(normalize=True).reset_index(name='count').pivot_table(
            index='split', columns='members', values='count').fillna(0)
        entropy_after = entropy(grouped_distrib, axis=1)
        entropy_after *= y.value_counts(sort=False, normalize=True)
        ig = entropy_before - entropy_after.sum()
        igResult.append(ig)
        count += 1
    InformartionGain = pd.Series(igResult)
    InformartionGain.index = columnName
    return InformartionGain.sort_values(ascending=False)

def convert_date(date_str):
    # Remove ordinal suffixes ('st', 'nd', 'rd', 'th') from the day
    date_str = date_str.strip()

    # Remove ordinal suffixes ('st', 'nd', 'rd', 'th') from the day
    date_str = re.sub(r'(\d+)(st|nd|rd|th)', r'\1', date_str)

    # Convert to datetime object
    return datetime.strptime(date_str, "%B %d, %Y")

def clean_year(value):
    # Check if the value contains a year range (e.g., "2015/2016" or "2009-2010")
    if isinstance(value, str) and ('/' in value or '-' in value):
        # Check if the first part is a valid 4-digit year
        first_part = value.split('/')[0].split('-')[0]
        if first_part.isdigit() and len(first_part) == 4:
            return int(first_part)
        else:
            # If not a valid year (e.g., "08/09"), return 0
            return 0
    # Check if the value is a valid 4-digit year
    elif isinstance(value, str) and value.isdigit() and len(value) == 4:
        return int(value)
    else:
        # If the value is not a valid year, return 0
        return 0

def convert_to_kg(weight_str):
    if isinstance(weight_str, str):
        weight_str = weight_str.lower().strip()
        if "kg" in weight_str:
            return float(weight_str.replace("kg", "").strip())
        elif "lbs" in weight_str:
            lbs_value = float(weight_str.replace("lbs", "").strip())
            return lbs_value * 0.453592  # Convert pounds to kg
    try:
        # If there is no unit, just return the numeric value without conversion
        return float(weight_str)
    except ValueError:
        return None  # If it's not a valid number, return None


def adjust_range_values(altitude):
    altitude = str(altitude).strip().lower()

    # Remove unwanted units and characters, including words and patterns like '..n..', 'dl', 'fee', etc.
    unwanted_chars = ['m', 'msnm', 'sn', 'eters', 'etros', 'metros', '~', '.s.l', 's', 'de', 'p', '公尺', 't', 'al',
                      'f.', 'fee', 'hru', 'dl', '..n..', '.']
    for char in unwanted_chars:
        altitude = altitude.replace(char, '')

    altitude = ' '.join(altitude.split())  # Normalize spaces

    # Handle ranges with 'a' as a separator (e.g., "1200 a 1400")
    if ' a ' in altitude:
        altitude = altitude.replace(' a ', '-')

    # Handle ranges with space or dash separators
    if ' ' in altitude or '-' in altitude:
        # Replace space with dash for consistent handling
        altitude = altitude.replace(' ', '-')
        parts = [float(x) for x in altitude.split('-') if
                 x.strip().replace('.', '').isdigit()]  # Skip non-numeric parts
        if len(parts) == 2:
            return round(sum(parts) / 2)  # Return the average of the range, rounded

    # If it's a single value, return it, rounding if necessary
    if altitude.replace('.', '').isdigit():  # Check if the remaining string is numeric
        return round(float(altitude))

    return None  # Return None if the value can't be converted
def decode_column(encoded_column, column_name, encoders):
    return encoders[column_name].inverse_transform(encoded_column)

def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH / Path(path)

class main:
    def __init__(self):
        self.dataframe = pd.DataFrame()
        self.feature_importance = []

        self.window = Tk()
        self.window.geometry("1284x685")
        self.window.configure(bg="#EAD8C0")

        self.canvas = Canvas(
            self.window,
            bg="#EAD8C0",
            height=685,
            width=1284,
            bd=0,
            highlightthickness=0,
            relief="ridge"
        )

        self.canvas.place(x=0, y=0)
        # ================ Show Data =====================

        self.button_image_3 = PhotoImage(
            file=relative_to_assets("button_3.png"))

        self.button_show_data = Button(
            image=self.button_image_3,
            borderwidth=0,
            highlightthickness=0,
            command=lambda: self.event_button_show_data(),
            relief="flat"
        )
        self.button_show_data.place(
            x=27.0,
            y=100.0,
            width=80.0,
            height=20.0
        )

        self.image_image_3 = PhotoImage(
            file=relative_to_assets("image_3.png"))
        image_3 = self.canvas.create_image(
            319.0,
            247.0,
            image=self.image_image_3
        )

        self.image_image_4 = PhotoImage(
            file=relative_to_assets("image_4.png"))
        image_4 = self.canvas.create_image(
            640.0,
            40.0,
            image=self.image_image_4
        )

        self.canvas.create_text(
            644.0,
            18.0,
            anchor="nw",
            text="Pendeteksi Varietas Kopi Arabika",
            fill="#FFF2E1",
            font=("Inter ExtraBold", 36 * -1)
        )

        self.image_image_5 = PhotoImage(
            file=relative_to_assets("image_5.png"))
        image_5 = self.canvas.create_image(
            47.0,
            39.0,
            image=self.image_image_5
        )

        self.image_image_6 = PhotoImage(
            file=relative_to_assets("image_6.png"))
        image_6 = self.canvas.create_image(
            1246.0,
            25.0,
            image=self.image_image_6
        )

        self.canvas.create_text(
            913.0,
            98.0,
            anchor="nw",
            text="Information Gain",
            fill="#000000",
            font=("Inter Bold", 12 * -1)
        )
# ================ Pre Process =====================
        self.image_image_2 = PhotoImage(
            file=relative_to_assets("image_2.png"))
        image_2 = self.canvas.create_image(
            961.0,
            247.0,
            image=self.image_image_2
        )

        self.button_image_2 = PhotoImage(
            file=relative_to_assets("button_2.png"))
        self.button_preprocess = Button(
            image=self.button_image_2,
            borderwidth=0,
            highlightthickness=0,
            command=lambda: self.event_button_preproses(),
            relief="flat"
        )
        self.button_preprocess.place(
            x=27.0,
            y=385.0,
            width=108.0,
            height=20.0
        )

# ================ Clasification =====================
        self.button_image_4 = PhotoImage(
            file=relative_to_assets("button_4.png"))
        self.button_submit = Button(
            image=self.button_image_4,
            borderwidth=0,
            highlightthickness=0,
            command=lambda: self.event_button_clasify(),
            relief="flat"
        )

        self.button_submit.place(
            x=1174.0,
            y=367.0,
            width=80.0,
            height=20.0
        )

        self.entry_image_32 = PhotoImage(
            file=relative_to_assets("entry_32.png"))
        entry_bg_32 = self.canvas.create_image(
            1181.4321060180664,
            421.0,
            image=self.entry_image_32
        )
        self.field_akurasi = Entry(
            bd=0,
            bg="#A79277",
            fg="#000716",
            highlightthickness=0
        )
        self.field_akurasi.place(
            x=1110.0,
            y=397.0,
            width=142.8642120361328,
            height=46.0
        )

        self.entry_image_33 = PhotoImage(
            file=relative_to_assets("entry_33.png"))
        entry_bg_33 = self.canvas.create_image(
            812.0,
            407.0,
            image=self.entry_image_33
        )
        self.input_cv = Entry(
            bd=0,
            bg="#D1BB9E",
            fg="#000716",
            highlightthickness=0
        )
        self.input_cv.place(
            x=790.0,
            y=397.0,
            width=44.0,
            height=18.0
        )

        self.canvas.create_text(
            669.0,
            399.0,
            anchor="nw",
            text="Cross Validation",
            fill="#000000",
            font=("Inter Bold", 12 * -1)
        )

        self.entry_image_34 = PhotoImage(
            file=relative_to_assets("entry_34.png"))
        entry_bg_34 = self.canvas.create_image(
            812.0,
            380.0,
            image=self.entry_image_34
        )
        self.input_n_tree = Entry(
            bd=0,
            bg="#D1BB9E",
            fg="#000716",
            highlightthickness=0
        )
        self.input_n_tree.place(
            x=790.0,
            y=370.0,
            width=44.0,
            height=18.0
        )

        self.canvas.create_text(
            669.0,
            372.0,
            anchor="nw",
            text="Tree",
            fill="#000000",
            font=("Inter Bold", 12 * -1)
        )

        self.canvas.create_text(
            887.0,
            437.0,
            anchor="nw",
            text="Pengujian Data Tunggal",
            fill="#000000",
            font=("Inter Bold", 12 * -1)
        )

        # hasil pred
        self.entry_image_1 = PhotoImage(
            file=relative_to_assets("entry_1.png"))
        entry_bg_1 = self.canvas.create_image(
            1177.4321060180664,
            579.0,
            image=self.entry_image_1
        )

        self.field_hasil_prediksi = Entry(
            bd=0,
            bg="#A79277",
            fg="#000716",
            highlightthickness=0
        )
        self.field_hasil_prediksi.place(
            x=1106.0,
            y=555.0,
            width=142.8642120361328,
            height=46.0
        )
        # button pred
        self.button_image_1 = PhotoImage(
            file=relative_to_assets("button_1.png"))
        self.button_predict = Button(
            image=self.button_image_1,
            borderwidth=0,
            highlightthickness=0,
            command=lambda: self.event_button_predict(),
            relief="flat"
        )
        self.button_predict.place(
            x=1174.0,
            y=513.0,
            width=80.0,
            height=20.0
        )
        self.window.resizable(False, False)
        self.window.mainloop()

    def event_button_show_data(self):
        global output_path
        file = filedialog.askopenfile()
        if file is None:  # User cancelled the dialog
            return

        # Safely get the file path
        output_path = file.name

        self.data = pd.read_csv(output_path)

        table_columns = list(self.data.columns.values)
        table_data = self.data.to_numpy().tolist()

        tree_vertical_scroll = ttk.Scrollbar(self.canvas, orient="vertical")
        tree_vertical_scroll.place(x=583, y=152, height=193, width=15)
        tree_horizontal_scroll = ttk.Scrollbar(self.canvas, orient="horizontal")
        tree_horizontal_scroll.place(x=40, y=330, height=15, width=543)

        table = ttk.Treeview(self.canvas, columns=table_columns, show='headings', yscrollcommand = tree_vertical_scroll,
                             xscrollcommand=tree_horizontal_scroll)
        table.place(x=40, y=152, height=178, width=543)

        for i in table_columns:
            table.column(i)
            table.heading(i, text=i)
        for dt in table_data:
            v = [r for r in dt]
            table.insert('', 'end', iid=v[0], values=v)

        tree_vertical_scroll.config(command=table.yview)
        tree_horizontal_scroll.config(command=table.xview)

        style = ttk.Style()
        style.theme_use('default')
        style.configure("Treeview", background='#D1BB9E', foreground='black')
        style.configure("Treeview.Heading", background='#A79277', foreground='black')
        style.map("Treeview", background=[("selected","#A79277")])

        style.configure('Horizontal.TScrollbar', background="#D1BB9E")
        style.configure('Vertical.TScrollbar', background="#D1BB9E")

    def event_button_preproses(self):
        try:
            df = self.data
            df = df[df['Variety'].isin(['Bourbon', 'Caturra', 'Typica'])]

            # Fill missing values with appropriate values
            # For numerical columns, you can use mean/median
            for col in df.select_dtypes(include=['float64', 'int64']).columns:
                df[col] = df[col].fillna(df[col].median())


            # For categorical columns, you can use the mode
            for col in df.select_dtypes(include=['object']).columns:
                df[col] = df[col].fillna(df[col].mode()[0])


            # drop unnecessary column
            df.drop(columns=['ID', 'Species'], inplace=True)

            # Check for duplicates and reset index
            df.drop_duplicates(inplace=True)
            df = df.reset_index(drop=True)

            # --data transformation--
            # convert measurement ft to m
            count = 0
            temp_unit_measeurement = df['unit_of_measurement'].tolist()
            temp_altitude_low_meters = df['altitude_low_meters'].tolist()
            temp_altitude_high_meters = df['altitude_high_meters'].tolist()
            temp_altitude_mean_meters = df['altitude_mean_meters'].tolist()

            for row in df['unit_of_measurement']:
                if row == 'ft':
                    temp_altitude_low_meters[count] = temp_altitude_low_meters[count] * 0.3048
                    temp_altitude_high_meters[count] = temp_altitude_high_meters[count] * 0.3048
                    temp_altitude_mean_meters[count] = temp_altitude_mean_meters[count] * 0.3048
                count = count + 1

            df['altitude_low_meters'] = pd.DataFrame(temp_altitude_low_meters)
            df['altitude_high_meters'] = pd.DataFrame(temp_altitude_high_meters)
            df['altitude_mean_meters'] = pd.DataFrame(temp_altitude_mean_meters)

            df.drop(columns=['unit_of_measurement'], inplace=True)  # unit_of_measurement unnecessary now

            # split numeric and categorical
            numeric_column = ['Altitude', 'Number.of.Bags', 'Bag.Weight', 'Aroma', 'Flavor', 'Aftertaste', 'Acidity',
                              'Body', 'Balance',
                              'Uniformity', 'Clean.Cup', 'Sweetness', 'Cupper.Points', 'Total.Cup.Points', 'Moisture',
                              'Category.One.Defects',
                              'Quakers', 'Category.Two.Defects', 'altitude_low_meters', 'altitude_high_meters',
                              'altitude_mean_meters']
            date_column = ['Harvest.Year', 'Grading.Date', 'Expiration']

            numeric_featrues = df[numeric_column]
            date_column = df[date_column]
            categorical_features = df.drop(columns=numeric_column)
            categorical_features = categorical_features.drop(columns=date_column)

            # Transform Categorical from word to number
            self.encoders = {}
            for column in categorical_features:
                self.encoders[column] = LabelEncoder()
                df[column] = self.encoders[column].fit_transform(df[column])

            # handle date type
            df_temp = date_column['Grading.Date']
            attribute_list = df_temp.tolist()
            replace = []
            count = 0
            for loop in range(len(attribute_list)):
                date_text = str(attribute_list[count])
                date_obj = convert_date(date_text)
                replace.insert(loop, date_obj)
                count = count + 1
            df['Grading.Date'] = pd.DataFrame(replace)
            df['Grading.Date_year'] = df['Grading.Date'].dt.year
            df['Grading.Date_month'] = df['Grading.Date'].dt.month
            df['Grading.Date_day'] = df['Grading.Date'].dt.day

            df_temp = df['Expiration']
            attribute_list = df_temp.tolist()
            replace = []
            count = 0
            for loop in range(len(attribute_list)):
                date_text = str(attribute_list[count])
                date_obj = convert_date(date_text)
                replace.insert(loop, date_obj)
                count = count + 1

            df['Expiration'] = pd.DataFrame(replace)
            df['Expiration_year'] = df['Expiration'].dt.year
            df['Expiration_month'] = df['Expiration'].dt.month
            df['Expiration_day'] = df['Expiration'].dt.day

            df = df.drop(columns=['Grading.Date', 'Expiration'])

            # handle harvest year
            df['Harvest.Year'] = df['Harvest.Year'].apply(clean_year)

            # handle bag weiht
            df['Bag.Weight'] = df['Bag.Weight'].apply(convert_to_kg)

            # handle allttitude
            df['Altitude'] = df['Altitude'].apply(adjust_range_values)

            scaler = MinMaxScaler()
            df[numeric_column] = scaler.fit_transform(df[numeric_column])
            self.x_min = scaler.data_min_
            self.x_max = scaler.data_max_

            #------------------------------------------------------------------------------------

            self.dataframe = df
            table_columns = list(df.columns.values)
            table_data = df.to_numpy().tolist()

            tree_vertical_scroll = ttk.Scrollbar(self.canvas, orient="vertical")
            tree_vertical_scroll.place(x=583, y=439, height=193, width=15)
            tree_horizontal_scroll = ttk.Scrollbar(self.canvas, orient="horizontal")
            tree_horizontal_scroll.place(x=40, y=617, height=15, width=543)

            table = ttk.Treeview(self.canvas, columns=table_columns, show='headings',
                                 yscrollcommand=tree_vertical_scroll,
                                 xscrollcommand=tree_horizontal_scroll)
            table.place(x=40, y=439, height=178, width=543)

            for i in table_columns:
                table.column(i)
                table.heading(i, text=i)
            for dt in table_data:
                v = [r for r in dt]
                table.insert('', 'end', values=v)

            tree_vertical_scroll.config(command=table.yview)
            tree_horizontal_scroll.config(command=table.xview)

            style = ttk.Style()
            style.theme_use('default')
            style.configure("Treeview", background='#D1BB9E', foreground='black')
            style.configure("Treeview.Heading", background='#A79277', foreground='black')
            style.map("Treeview", background=[("selected", "#A79277")])

            style.configure('Horizontal.TScrollbar', background="#D1BB9E")
            style.configure('Vertical.TScrollbar', background="#D1BB9E")

            # ------------------------------------------------------------------------------------
            X = df.drop(columns=['Variety'])
            y = df['Variety']

            infgain = information_gain(X, y)
            infGainIndex = infgain.index
            temp_inf_gain_index = []

            rank = 10
            for i in range(rank):
                temp_inf_gain_index.append(infGainIndex[i])
            self.feature_importance = temp_inf_gain_index
            info_gain = pd.DataFrame({'Information gain': self.feature_importance})


            table_columns = list(info_gain.columns.values)
            table_data = info_gain.to_numpy().tolist()

            tree_vertical_scroll = ttk.Scrollbar(self.canvas, orient="vertical")
            tree_vertical_scroll.place(x=1227, y=157, height=193, width=15)
            tree_horizontal_scroll = ttk.Scrollbar(self.canvas, orient="horizontal")
            tree_horizontal_scroll.place(x=695, y=327, height=15, width=543)

            table = ttk.Treeview(self.canvas, columns=table_columns, show='headings',
                                 yscrollcommand=tree_vertical_scroll,
                                 xscrollcommand=tree_horizontal_scroll)
            table.place(x=695, y=159, height=178, width=543)

            for i in table_columns:
                table.column(i)
                table.heading(i, text=i)
            for dt in table_data:
                v = [r for r in dt]
                table.insert('', 'end', values=v)

            tree_vertical_scroll.config(command=table.yview)
            tree_horizontal_scroll.config(command=table.xview)

            style = ttk.Style()
            style.theme_use('default')
            style.configure("Treeview", background='#D1BB9E', foreground='black')
            style.configure("Treeview.Heading", background='#A79277', foreground='black')
            style.map("Treeview", background=[("selected", "#A79277")])

            style.configure('Horizontal.TScrollbar', background="#D1BB9E")
            style.configure('Vertical.TScrollbar', background="#D1BB9E")

            text_position = [[674.0,511.0],
                             [674.0,538.0],
                             [674.0,566.0],
                             [674.0,599.0],
                             [674.0,626.0],
                             [866.0,511.0],
                             [866.0,538.0],
                             [866.0,566.0],
                             [866.0,599.0],
                             [866.0,626.0]]
            field_position = [[760.0,511.0],
                              [760.0,538.0],
                              [760.0,566.0],
                              [760.0,599.0],
                              [760.0,626.0],
                              [990.0,511.0],
                              [990.0,538.0],
                              [990.0,566.0],
                              [990.0,599.0],
                              [990.0,626.0]]

            i = 0
            decode_df = df.copy()

            # Decode categorical features for proper representation in Combobox
            for column in categorical_features:
                decode_df[column] = decode_column(decode_df[column], column, self.encoders)

            # Dynamically create input fields for feature importance
            for feature in self.feature_importance:
                self.canvas.create_text(
                    text_position[i][0],
                    text_position[i][1],
                    anchor="nw",
                    text=feature,
                    fill="#000000",
                    font=("Inter Bold", 12 * -1)
                )

                if feature in numeric_column or feature in date_column:
                    setattr(self, f"input_attribut_{i+1}", Entry(
                        bd=0,
                        bg="#D1BB9E",
                        fg="#000716",
                        highlightthickness=0
                    ))
                    getattr(self, f"input_attribut_{i+1}").place(
                        x=field_position[i][0],
                        y=field_position[i][1],
                        width=44.0,
                        height=18.0
                    )
                else:
                    options = decode_df[feature].unique().tolist()
                    setattr(self, f"input_attribut_{i+1}", ttk.Combobox(
                        state="readonly",
                        values=options
                    ))
                    getattr(self, f"input_attribut_{i+1}").place(
                        x=field_position[i][0],
                        y=field_position[i][1],
                        width=100.0,
                        height=18.0
                    )

                i += 1
        except AttributeError:
            messagebox.showerror("showerror",
                                 "Harap Import Dataset Terlebih Dahulu")
        except ValueError:
            messagebox.showerror("showerror",
                                 "Harap Import Dataset Terlebih Dahulu")

    def event_button_clasify(self):
        try:
            n_estimator = int(self.input_n_tree.get())
            fold = int(self.input_cv.get())

            x = self.dataframe[self.feature_importance]
            y = self.dataframe['Variety']

            X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.015, random_state=42)

            testing_data = X_test
            testing_data['Variety'] = y_test
            decode_testing_data = testing_data.copy()
            for column in testing_data.columns:
                decode_testing_data[column] = decode_column(decode_testing_data[column], column, self.encoders)

            #print(tabulate(decode_testing_data, headers='keys'))
            with pd.ExcelWriter('Testing_Data.xlsx') as writer:
                decode_testing_data.to_excel(writer, sheet_name='Testing_Data')


            ASM_function = ['entropy', 'gini']

            start_time = time.time()

            model_RDF = RandomForestClassifier(criterion=ASM_function[0], n_estimators=n_estimator)
            model_RDF.fit(X_train, y_train)
            cv_result = cross_val_score(model_RDF, X_train, y_train, cv=fold, scoring='accuracy')
            self.trained_model = model_RDF

            end_time = time.time()
            execution_time = end_time - start_time
            print(f"Execution time: {execution_time} seconds")

            akurasi = round(cv_result.mean(), 4)
            akurasi = akurasi * 100
            text = "Akurasi : " + str(akurasi) + " %"

            self.field_akurasi.delete(0, 'end')
            self.field_akurasi.insert(0, text)
        except AttributeError:
            messagebox.showerror("showerror",
                                 "Harap Import Dataset Terlebih Dahulu Dan Lakukan Pre-Processed Data")
        except ValueError:
            messagebox.showerror("showerror",
                                 "Input Harus Di Isi Dahulu !!!\n "
                                 "Harus Berisi Angka !!!\n")

    def event_button_predict(self):

            input_value = [[self.input_attribut_1.get(),
                           self.input_attribut_2.get(),
                           self.input_attribut_3.get(),
                           self.input_attribut_4.get(),
                           self.input_attribut_5.get(),
                           self.input_attribut_6.get(),
                           self.input_attribut_7.get(),
                           self.input_attribut_8.get(),
                           self.input_attribut_9.get(),
                           self.input_attribut_10.get()]]

            df = pd.DataFrame(input_value, columns=[self.feature_importance[0],
                                                    self.feature_importance[1],
                                                    self.feature_importance[2],
                                                    self.feature_importance[3],
                                                    self.feature_importance[4],
                                                    self.feature_importance[5],
                                                    self.feature_importance[6],
                                                    self.feature_importance[7],
                                                    self.feature_importance[8],
                                                    self.feature_importance[9]])
            for column in self.feature_importance:
                df[column] = self.encoders[column].transform(df[column])
            predictions = self.trained_model.predict(df)
            predictions = decode_column(predictions, 'Variety', self.encoders)
            text = str(predictions[0])

            self.field_hasil_prediksi.delete(0, 'end')
            self.field_hasil_prediksi.insert(0, text)




'''
        except ValueError:
            messagebox.showerror("showerror",
                                 "Input Salah Atau Belum Di Isi Semua !!!\n "
                                 "Harus Berisi Angka dan Harus Bernilai 0 - 10\n"
                                 "Jika Angka Bernilai Koma Gunakan (.) cth 9.5")
'''
if __name__ == "__main__":
    main()

