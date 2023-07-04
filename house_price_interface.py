import logging
import logging.config
import pickle
import timeit

import wx
import pandas as pd
import pipeline
from datetime import datetime


class CustomConsoleHandler(logging.StreamHandler):
    """
    Custom console handler for redirecting logging messages to a text control.

    This class extends the `logging.StreamHandler` class to redirect logging
    messages to a UI element, represented by the `textctrl` parameter.

    Args:
        textctrl (object): The UI element where the logging messages will be displayed.

    Attributes:
        textctrl (object): The UI element where the logging messages will be displayed.

    Usage:
        Instantiate an object of the `CustomConsoleHandler` class and pass the
        `textctrl` object as the parameter. This will redirect logging messages to
        the specified UI element.

    Example:
        text_ctrl = wx.TextCtrl(parent, style=wx.TE_MULTILINE)
        console_handler = CustomConsoleHandler(text_ctrl)
        logger.addHandler(console_handler)
    """

    # ----------------------------------------------------------------------
    def __init__(self, textctrl):
        """
        Initialize the CustomConsoleHandler object.

        Args:
            textctrl (object): The UI element where the logging messages will be displayed.
        """
        logging.StreamHandler.__init__(self)
        self.textctrl = textctrl


    def emit(self, record):
        """
        Emit the logging record to the text control.

        This method is called by the logging system to handle a logging record.
        It formats the record into a message using the defined format and writes
        it to the associated text control.

        Args:
            record (logging.LogRecord): The logging record to be emitted.

        Returns:
            None

        Raises:
            None

        Example:
            # Assuming 'logger' is an instance of the logger that uses the CustomConsoleHandler
            logger.debug("This is a debug message")
        """
        msg = self.format(record)
        self.textctrl.WriteText(msg + "\n")
        self.flush()


class LoadModelPanel(wx.Panel):
    """
    A panel for loading and training machine learning models.

    This panel provides options for loading an existing model or training a new model.
    It includes input fields for specifying the file name of the model to load and the state abbreviation
    for training data. It also displays a log of events using a text control.

    Args:
        parent (wx.Window): The parent window that contains this panel.

    Attributes:
        file_name_input (wx.TextCtrl): The input field for specifying the file name of the model to load.
        load_button (wx.Button): The button for triggering the load action.
        state_input (wx.TextCtrl): The input field for specifying the state abbreviation for training data.
        train_button (wx.Button): The button for triggering the train action.
        train_file_out_input (wx.TextCtrl): The input field for specifying the name of the saved model.
        logger (logging.Logger): The logger instance for logging events.
    """
    def __init__(self, parent):
        """
        Initialize the LoadModelPanel.

        Args:
            parent (wx.Window): The parent window that contains this panel.
        """
        super().__init__(parent)
        # self.data_instance = march_madness.load_model()

        load_label = wx.StaticText(self, label="Would you like to load a model? What is the file name of the model?")
        train_label = wx.StaticText(self,
                                    label="Would you like to train a new model? \nWhat is the State abbreviation you "
                                          "want to pull training data from? eg. MO")
        train_file_out_label = wx.StaticText(self,
                                    label="What would you like to name the saved model?")
        self.logger = logging.getLogger("wxApp")

        logText = wx.TextCtrl(self,
                              style=wx.TE_MULTILINE | wx.TE_READONLY | wx.HSCROLL, size=(500, 400))
        self.state_input = wx.TextCtrl(self, style=wx.TE_PROCESS_ENTER)
        self.file_name_input = wx.TextCtrl(self, style=wx.TE_PROCESS_ENTER)
        self.train_file_out_input = wx.TextCtrl(self, style=wx.TE_PROCESS_ENTER)
        self.load_button = wx.Button(self, label='Load')
        self.grid_search_toggle = wx.CheckBox(self, label="Grid Search", style=wx.BU_AUTODRAW)
        self.train_button = wx.Button(self, label='Train')
        # self.result_txt = wx.TextCtrl(self, size=(500, 400))

        load_sizer_horizontal = wx.BoxSizer(wx.HORIZONTAL)
        train_sizer_vertical = wx.BoxSizer(wx.VERTICAL)
        train_file_out_sizer_horizontal = wx.BoxSizer(wx.HORIZONTAL)
        train_sizer = wx.BoxSizer(wx.VERTICAL)
        load_sizer_horizontal.Add(self.file_name_input, flag=wx.ALL | wx.LEFT, border=5)
        load_sizer_horizontal.Add(self.load_button, flag=wx.ALL | wx.LEFT, border=5)
        train_file_out_sizer_horizontal.Add(self.train_file_out_input, flag=wx.ALL | wx.LEFT, border=5)
        train_file_out_sizer_horizontal.Add(self.grid_search_toggle, flag=wx.ALL | wx.LEFT, border=5)
        train_file_out_sizer_horizontal.Add(self.train_button, flag=wx.ALL | wx.LEFT, border=5)
        train_sizer.Add(train_label, flag=wx.ALL | wx.LEFT, border=5)
        train_sizer.Add(self.state_input, flag=wx.ALL | wx.LEFT, border=5)
        train_sizer.Add(train_file_out_label, flag=wx.ALL | wx.LEFT, border=5)
        train_sizer.Add(train_file_out_sizer_horizontal, flag=wx.ALL | wx.LEFT, border=5)
        # teams_sizer.Add(input_sizer1)
        main_sizer = wx.BoxSizer(wx.VERTICAL)
        main_sizer.Add(load_label, proportion=1, flag=wx.ALL | wx.LEFT, border=5)
        main_sizer.Add(load_sizer_horizontal, proportion=1, flag=wx.ALL | wx.LEFT, border=5)
        # main_sizer.Add(train_label, proportion=.5, flag=wx.ALL | wx.LEFT, border=5)
        main_sizer.Add(train_sizer, proportion=1, flag=wx.ALL | wx.LEFT, border=5)
        # main_sizer.Add(button, proportion=.5, flag=wx.ALL | wx.CENTER, border=5)
        # main_sizer.Add(self.result_txt, proportion=.5, flag=wx.ALL | wx.CENTER, border=5)
        main_sizer.Add(logText, proportion=1, flag=wx.ALL | wx.CENTER, border=5)

        self.SetSizer(main_sizer)

        txtHandler = CustomConsoleHandler(logText)
        self.logger.addHandler(txtHandler)


class MenuPanel(wx.Panel):

    def __init__(self, parent):
        super().__init__(parent)
        find_house_label = wx.StaticText(self, label="Find specific house")
        filter_df_label = wx.StaticText(self, label="Create filtered csv file")
        self.find_house_button = wx.Button(self, label='Choose')
        self.filter_df_button = wx.Button(self, label='Choose')

        find_house_sizer_horizontal = wx.BoxSizer(wx.HORIZONTAL)
        filter_df_sizer_horizontal = wx.BoxSizer(wx.HORIZONTAL)
        find_house_sizer_horizontal.Add(self.find_house_button, flag=wx.ALL | wx.CENTER, border=5)
        find_house_sizer_horizontal.Add(find_house_label, flag=wx.ALL | wx.CENTER, border=5)
        filter_df_sizer_horizontal.Add(self.filter_df_button, flag=wx.ALL | wx.CENTER, border=5)
        filter_df_sizer_horizontal.Add(filter_df_label, flag=wx.ALL | wx.CENTER, border=5)
        main_sizer = wx.BoxSizer(wx.VERTICAL)
        main_sizer.Add(find_house_sizer_horizontal, proportion=1, flag=wx.ALL | wx.LEFT, border=5)
        main_sizer.Add(filter_df_sizer_horizontal, proportion=10, flag=wx.ALL | wx.LEFT, border=5)

        self.SetSizer(main_sizer)


class FindHousePanel(wx.Panel):
    """
    A panel for displaying menu options.

    This panel provides options for finding a specific house and creating a filtered CSV file.
    It includes buttons for each option and corresponding labels.

    Args:
        parent (wx.Window): The parent window that contains this panel.

    Attributes:
        find_house_button (wx.Button): The button for choosing the find specific house option.
        filter_df_button (wx.Button): The button for choosing the create filtered CSV file option.
    """
    def __init__(self, parent):
        """
        Initialize the MenuPanel.

        Args:
            parent (wx.Window): The parent window that contains this panel.
        """
        super().__init__(parent)

        self.zip_input = wx.TextCtrl(self, style=wx.TE_PROCESS_ENTER)
        zip_label = wx.StaticText(self, label="Enter Zip code of house")
        house_number_label = wx.StaticText(self, label="Enter house number of house")
        self.house_number_input = wx.TextCtrl(self, style=wx.TE_PROCESS_ENTER)
        self.submit_button = wx.Button(self, label='Submit')
        self.result_txt = wx.TextCtrl(self, size=(500, 100), style=wx.TE_MULTILINE)

        zip_sizer = wx.BoxSizer(wx.HORIZONTAL)
        zip_sizer.Add(self.zip_input, flag=wx.ALL | wx.LEFT, border=5)
        zip_sizer.Add(zip_label, flag=wx.ALL | wx.LEFT, border=5)

        house_number_sizer = wx.BoxSizer(wx.HORIZONTAL)
        house_number_sizer.Add(self.house_number_input, flag=wx.ALL | wx.LEFT, border=5)
        house_number_sizer.Add(house_number_label, flag=wx.ALL | wx.LEFT, border=5)

        main_sizer = wx.BoxSizer(wx.VERTICAL)
        main_sizer.Add(zip_sizer, proportion=1, flag=wx.ALL | wx.ALIGN_LEFT, border=5)
        main_sizer.Add(house_number_sizer, proportion=1, flag=wx.ALL | wx.ALIGN_LEFT, border=5)
        main_sizer.Add(self.submit_button, proportion=1, flag=wx.ALL | wx.ALIGN_LEFT, border=5)
        main_sizer.Add(self.result_txt, proportion=1, flag=wx.ALL | wx.ALIGN_LEFT, border=5)
        self.SetSizer(main_sizer)


class FilterTablePanel(wx.Panel):
    """
    A panel for filtering table data.

    This panel provides input fields for specifying filter criteria to generate a filtered CSV file.
    It includes input fields for minimum number of bedrooms, minimum square footage, maximum list price,
    counties (separated by comma), state abbreviation, and the desired file name for the generated CSV file.

    Args:
        parent (wx.Window): The parent window that contains this panel.

    Attributes:
        beds_input (wx.TextCtrl): The input field for the minimum number of bedrooms.
        sqft_input (wx.TextCtrl): The input field for the minimum square footage.
        price_input (wx.TextCtrl): The input field for the maximum list price.
        counties_input (wx.TextCtrl): The input field for entering counties (separated by comma).
        state_input (wx.TextCtrl): The input field for entering the state abbreviation.
        file_out_input (wx.TextCtrl): The input field for specifying the file name for the generated CSV file.
        submit_button (wx.Button): The button for submitting the filter criteria.
        result_txt (wx.TextCtrl): The text field for displaying the result.
    """
    def __init__(self, parent):
        """
        Initialize the FilterTablePanel.

        Args:
            parent (wx.Window): The parent window that contains this panel.
        """
        super().__init__(parent)
        self.beds_input = wx.TextCtrl(self, style=wx.TE_PROCESS_ENTER)
        beds_label = wx.StaticText(self, label="Enter minimum number of bedrooms in house")
        sqft_label = wx.StaticText(self, label="Enter minimum square footage of house")
        self.sqft_input = wx.TextCtrl(self, style=wx.TE_PROCESS_ENTER)
        self.price_input = wx.TextCtrl(self, style=wx.TE_PROCESS_ENTER)
        price_label = wx.StaticText(self, label="Enter maximum list price for house")
        self.counties_input = wx.TextCtrl(self, style=wx.TE_PROCESS_ENTER)
        counties_label = wx.StaticText(self, label="Enter counties separated by comma. eg. St. Louis, Jefferson")
        self.state_input = wx.TextCtrl(self, style=wx.TE_PROCESS_ENTER)
        state_label = wx.StaticText(self, label="What is the state abbreviation these counties are in? eg. MO")
        self.file_out_input = wx.TextCtrl(self, style=wx.TE_PROCESS_ENTER)
        file_out_label = wx.StaticText(self, label="What do you want the generated csv file to be called?")
        self.submit_button = wx.Button(self, label='Submit')
        self.result_txt = wx.TextCtrl(self, size=(500, 100))

        beds_sizer = wx.BoxSizer(wx.HORIZONTAL)
        beds_sizer.Add(self.beds_input, flag=wx.ALL | wx.LEFT, border=5)
        beds_sizer.Add(beds_label, flag=wx.ALL | wx.LEFT, border=5)

        sqft_sizer = wx.BoxSizer(wx.HORIZONTAL)
        sqft_sizer.Add(self.sqft_input, flag=wx.ALL | wx.LEFT, border=5)
        sqft_sizer.Add(sqft_label, flag=wx.ALL | wx.LEFT, border=5)

        price_sizer = wx.BoxSizer(wx.HORIZONTAL)
        price_sizer.Add(self.price_input, flag=wx.ALL | wx.LEFT, border=5)
        price_sizer.Add(price_label, flag=wx.ALL | wx.LEFT, border=5)

        counties_sizer = wx.BoxSizer(wx.HORIZONTAL)
        counties_sizer.Add(self.counties_input, flag=wx.ALL | wx.LEFT, border=5)
        counties_sizer.Add(counties_label, flag=wx.ALL | wx.LEFT, border=5)

        state_sizer = wx.BoxSizer(wx.HORIZONTAL)
        state_sizer.Add(self.state_input, flag=wx.ALL | wx.LEFT, border=5)
        state_sizer.Add(state_label, flag=wx.ALL | wx.LEFT, border=5)

        file_out_sizer = wx.BoxSizer(wx.HORIZONTAL)
        file_out_sizer.Add(self.file_out_input, flag=wx.ALL | wx.LEFT, border=5)
        file_out_sizer.Add(file_out_label, flag=wx.ALL | wx.LEFT, border=5)

        main_sizer = wx.BoxSizer(wx.VERTICAL)
        main_sizer.Add(beds_sizer, proportion=1, flag=wx.ALL | wx.LEFT, border=5)
        main_sizer.Add(sqft_sizer, proportion=1, flag=wx.ALL | wx.LEFT, border=5)
        main_sizer.Add(price_sizer, proportion=1, flag=wx.ALL | wx.LEFT, border=5)
        main_sizer.Add(counties_sizer, proportion=1, flag=wx.ALL | wx.LEFT, border=5)
        main_sizer.Add(state_sizer, proportion=1, flag=wx.ALL | wx.LEFT, border=5)
        main_sizer.Add(file_out_sizer, proportion=1, flag=wx.ALL | wx.LEFT, border=5)
        main_sizer.Add(self.submit_button, proportion=1, flag=wx.ALL | wx.LEFT, border=5)
        main_sizer.Add(self.result_txt, proportion=1, flag=wx.ALL | wx.LEFT, border=5)
        self.SetSizer(main_sizer)


class MyFrame(wx.Frame):
    """
    The main frame of the house price prediction application.

    This frame serves as the container for various panels and controls the navigation between them.

    Attributes:
        load_panel (LoadModelPanel): The panel for loading or training a machine learning model.
        menu_panel (MenuPanel): The panel for choosing specific actions.
        find_house_panel (FindHousePanel): The panel for finding a specific house.
        filter_table_panel (FilterTablePanel): The panel for filtering table data.
    """
    def __init__(self):
        """
        Initialize the MyFrame.

        This method sets up the main frame, creates and adds the necessary panels, and sets up event handlers.

        """
        super().__init__(None, title="House Price Prediction")
        sizer = wx.BoxSizer()
        self.SetSizer(sizer)

        self.load_panel = LoadModelPanel(self)
        sizer.Add(self.load_panel, 1, wx.EXPAND)
        self.load_panel.load_button.Bind(wx.EVT_BUTTON, self.load)
        self.load_panel.train_button.Bind(wx.EVT_BUTTON, self.train)
        self.logger = logging.getLogger("wxApp")

        self.menu_panel = MenuPanel(self)
        sizer.Add(self.menu_panel, 1, wx.EXPAND)
        self.menu_panel.find_house_button.Bind(wx.EVT_BUTTON, self.to_find_house)
        self.menu_panel.filter_df_button.Bind(wx.EVT_BUTTON, self.to_filter_df)
        self.menu_panel.Hide()
        self.SetSize(800, 600)
        self.Center()

        self.find_house_panel = FindHousePanel(self)
        sizer.Add(self.find_house_panel, 1, wx.EXPAND)
        self.find_house_panel.Hide()
        self.find_house_panel.submit_button.Bind(wx.EVT_BUTTON, self.predict)
        self.SetSize(800, 600)
        self.Center()

        self.filter_table_panel = FilterTablePanel(self)
        sizer.Add(self.filter_table_panel, 1, wx.EXPAND)
        self.filter_table_panel.Hide()
        self.filter_table_panel.submit_button.Bind(wx.EVT_BUTTON, self.filter)
        self.SetSize(800, 600)
        self.Center()

    def load(self, event):
        """
        Load a machine learning model from a file.

        This method is triggered when the load button is clicked in the LoadModelPanel. It retrieves the file name
        from the input field, attempts to load the model from the file, and updates the necessary attributes
        in the frame. If the model is loaded successfully, the menu panel is shown and the load panel is hidden.

        Args:
            event: The event object triggered by clicking the load button.

        """
        file_name = self.load_panel.file_name_input.GetValue()

        try:
            t_section = timeit.default_timer()
            self.logger.info(datetime.now().strftime('%H:%M:%S.%f') + " - " + f"Starting to load model")
            self.r = pickle.load(open(file_name, 'rb'))
            self.regr_model = self.r.model
            self.logger.info(datetime.now().strftime(
                '%H:%M:%S.%f') + " - " + f"Model loaded")
            self.logger.info("Model loaded")
            self.menu_panel.Show()
            self.load_panel.Hide()
            self.Layout()
        except Exception as e:
            logging.info(datetime.now().strftime(
                '%H:%M:%S.%f') + " - " + f"File not found")
            self.logger.info(f"File not found because of {e}")

    def train(self, event):
        """
        Train a machine learning model.

        This method is triggered when the train button is clicked in the LoadModelPanel. It retrieves the state abbreviation
        and output file name from the input fields, scrapes the dataset using the Realtor API, builds a model based on the
        dataset, and evaluates the model's performance. If the model is built successfully, the menu panel is shown and the
        load panel is hidden.

        Args:
            event: The event object triggered by clicking the train button.

        """
        try:
            state_abbr = self.load_panel.state_input.GetValue()
            file_out = self.load_panel.train_file_out_input.GetValue()
            t_section = timeit.default_timer()
            self.logger.info(datetime.now().strftime('%H:%M:%S.%f') + " - " + f"Starting to scrape for dataset")
            self.r = pipeline.get_realtor_object(state_abbr)
            self.logger.info(datetime.now().strftime(
                '%H:%M:%S.%f') + " - " + f"Dataset scraped in {timeit.default_timer() - t_section} seconds")
            t_section = timeit.default_timer()
            self.logger.info(datetime.now().strftime('%H:%M:%S.%f') + " - " + f"Starting to build model")
            gs_toggle = self.load_panel.grid_search_toggle.GetValue()
            self.regr_model = pipeline.get_model(self.r, state_abbr, file_out, gs_toggle)
            pipeline.evaluate_model(self.regr_model)
            self.logger.info(datetime.now().strftime(
                '%H:%M:%S.%f') + " - " + f"Model built in {timeit.default_timer() - t_section} seconds")
            self.menu_panel.Show()
            self.load_panel.Hide()
            self.Layout()
        except Exception as e:
            self.logger.info(e)

    def to_find_house(self, event):
        """
        Switch to the FindHousePanel.

        This method is triggered when the find house button is clicked in the MenuPanel. It hides the menu panel and shows
        the FindHousePanel, allowing the user to find a specific house. The layout is updated accordingly.

        Args:
            event: The event object triggered by clicking the find house button.

        """
        self.menu_panel.Hide()
        self.find_house_panel.Show()
        self.Layout()

    def to_filter_df(self, event):
        """
        Switch to the FilterTablePanel.

        This method is triggered when the filter df button is clicked in the MenuPanel. It hides the menu panel and shows
        the FilterTablePanel, allowing the user to create a filtered CSV file. The layout is updated accordingly.

        Args:
            event: The event object triggered by clicking the filter df button.

        """
        self.menu_panel.Hide()
        self.filter_table_panel.Show()
        self.Layout()

    def predict(self, event):
        """
        Predict the price of a specific house.

        This method is triggered when the submit button is clicked in the FindHousePanel. It retrieves the zip code and
        house number entered by the user, and uses the trained model to predict the price of the specified house. The
        predicted price is compared with the actual price, if available, and the result is displayed in the result text
        control.

        Args:
            event: The event object triggered by clicking the submit button.

        """
        zip_code = self.find_house_panel.zip_input.GetValue()
        house_num = self.find_house_panel.house_number_input.GetValue()
        address_price = pipeline.predict_specific_address(self.r, self.regr_model, zip_code, house_num)
        try:
            if address_price[0] > address_price[1]:
                self.find_house_panel.result_txt.SetValue(
                    f"The model predicts a price of ${address_price[1]}. The actual price is ${address_price[0]}. The "
                    f"house is ${address_price[0] - address_price[1]} more expensive than the prediction.")
            elif address_price[0] < address_price[1]:
                self.find_house_panel.result_txt.SetValue(
                    f"The model predicts a price of ${address_price[1]}. The actual price is ${address_price[0]}. The "
                    f"house is ${address_price[1] - address_price[0]} cheaper than the prediction.")
            else:
                self.find_house_panel.result_txt.SetValue(f"The model predicts the exact price of ${address_price[0]}")
        except TypeError as e:
            logging.error(e)
            self.find_house_panel.result_txt.SetValue(address_price)

    def filter(self, event):
        """
        Filter houses based on specified criteria and create a CSV file.

        This method is triggered when the submit button is clicked in the FilterTablePanel. It retrieves the minimum number
        of bedrooms, minimum square footage, maximum price, counties, and state abbreviation entered by the user. It then
        calls the necessary functions to filter the houses based on the provided criteria and creates a CSV file with the
        filtered data.

        Args:
            event: The event object triggered by clicking the submit button.

        """
        min_beds = float(self.filter_table_panel.beds_input.GetValue())
        min_sqft = float(self.filter_table_panel.sqft_input.GetValue())
        max_price = float(self.filter_table_panel.price_input.GetValue())

        counties = list(map(str.strip, self.filter_table_panel.counties_input.GetValue().split(',')))
        state_abbr = self.filter_table_panel.state_input.GetValue()

        realtor_obj = pipeline.get_realtor_object(state_abbr)
        filtered_df = pipeline.find_deals(realtor_obj, self.regr_model, min_beds, min_sqft, max_price, counties, state_abbr)
        out_file = self.filter_table_panel.file_out_input.GetValue()
        filtered_df.to_csv(out_file)
        self.filter_table_panel.result_txt.SetValue(f"File created at {out_file}")


def main():
    """
    Entry point of the application.

    This function initializes the logging configuration, creates an instance of the wx.App class, creates an instance
    of the MyFrame class, and starts the main event loop.

    Returns:
        None

    """
    dictLogConfig = {
        "version": 1,
        "handlers": {
            "fileHandler": {
                "class": "logging.FileHandler",
                "formatter": "myFormatter",
                "filename": "test.log"
            },
            "consoleHandler": {
                "class": "logging.StreamHandler",
                "formatter": "myFormatter"
            }
        },
        "loggers": {
            "wxApp": {
                "handlers": ["fileHandler", "consoleHandler"],
                "level": "INFO",
            },
            "wxApp_find_house": {
                "handlers": ["fileHandler", "consoleHandler"],
                "level": "INFO",
            }
        },

        "formatters": {
            "myFormatter": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            }
        }
    }
    logging.config.dictConfig(dictLogConfig)
    logger = logging.getLogger("wxApp")

    app = wx.App(redirect=False)
    frame = MyFrame()
    frame.Show()
    app.MainLoop()


if __name__ == '__main__':
    main()
