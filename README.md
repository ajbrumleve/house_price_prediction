# house_price_prediction

<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a name="readme-top"></a>
<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->



<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]



<!-- PROJECT LOGO -->
<br />
<div align="center">

  <h3 align="center">House Price Prediction</h3>

  <p align="center">
    Python implementation scrape web for house price data, build a model, and predict housing prices.
    <br />
    <a href="https://github.com/ajbrumleve/house_price_prediction/issues">Report Bug</a>
    ·
    <a href="https://github.com/ajbrumleve/house_price_prediction">Request Feature</a>
  </p>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project


This was an attempt to solve a real world issue we are facing as we look to buy a house. I wanted a way to get a sense on if a house is overpriced or underpriced. 

This code scrapes Realtor.com for all of the houses for sale in a given state. It then trains a Random Forest Regressor to try to assign a price to a house based on a series of features.

I also designed a basic interface to train or load a model and use it to either check specific addresses or to create a filtered csv based on some criteria sorted by how underpriced a house is in the given counties.


<p align="right">(<a href="#readme-top">back to top</a>)</p>



### Built With

* [![Python][Python]][Python-url]


<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- GETTING STARTED -->
## Getting Started



### Prerequisites

Require packages kneed, scikit-learn, and numpy.

Use `pip` to install the packages from PyPI:

```bash
pip install wxPython
pip install scikit-learn
pip install numpy
pip install requests
pip install scipy
pip install termcolor
pip install matplotlib
pip install seaborn
pip install beautifulsoup4
```


### Installation



1. Download and unzip [this entire repository from GitHub](https://github.com/ajbrumleve/house_price_prediction), either interactively, or by entering the following in your Terminal.
    ```bash
    git clone https://github.com/ajbrumleve/house_price_prediction.git
    ```
2. Navigate into the top directory of the repo on your machine
    ```bash
    cd house_price_prediction
    ```
3. Create a virtualenv and install the package dependencies. If you don't have `pipenv`, you can follow instructions [here](https://pipenv.pypa.io/en/latest/install/) for how to install.
    ```bash
    pipenv install
    ```
4. Run `house_price_interface.py` to run the graphical interface. For a console interface, run pipeline.py. 

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- USAGE EXAMPLES -->
## Usage

If using the GUI interface, enter a file location and the state abbreviation you are interested in and click train. The log window will show the steps currently running. Once the model is trained you will immediately be able to try the model.

If a model has already been trained, you may load the model through the interface. There are two prediction activities available. The first lets you enter a zip code and a house number and see the comparison between the predicted price and the list price of that address. If the house is not for sale, you should get an error.

You can also create a filtered csv. You input a minimum number of bedrooms, minimum square footage, and a maximum price along with the counties to search in. The code will create a csv with all of the houses in those counties for sale which meet your criteria sorted by how much cheaper than the prediction the house is.

<p align="right">(<a href="#readme-top">back to top</a>)</p>




<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

Some ideas of ways to extend this code include:
 - Add more options to filter the dataset
 - Allow some customization of which features are used when the model is trained.
 - Incorporate historical house pricing data. As of now only houses currently or sale are used to train the model. This allows for the model to be up to date with current market trends, but also limits the amount of data available.
 - Combine outside zip code level data with the realtor.com dataset.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTACT -->
## Contact

Andrew Brumleve - [@AndrewBrumleve](https://twitter.com/AndrewBrumleve) - ajbrumleve@gmail.com

Project Link: [https://github.com/house_price_prediction](https://github.com/house_price_prediction)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

* [README Template](https://github.com/othneildrew/Best-README-Template)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/ajbrumleve/house_price_prediction.svg?style=for-the-badge
[contributors-url]: https://github.com/ajbrumleve/house_price_prediction/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/ajbrumleve/house_price_prediction.svg?style=for-the-badge
[forks-url]: https://github.com/ajbrumleve/house_price_prediction/network/members
[stars-shield]: https://img.shields.io/github/stars/ajbrumleve/house_price_prediction.svg?style=for-the-badge
[stars-url]: https://github.com/ajbrumleve/house_price_prediction/stargazers
[issues-shield]: https://img.shields.io/github/issues/ajbrumleve/house_price_prediction.svg?style=for-the-badge
[issues-url]: https://github.com/ajbrumleve/house_price_prediction/issues
[license-shield]: https://img.shields.io/github/license/ajbrumleve/house_price_prediction.svg?style=for-the-badge
[license-url]: https://github.com/ajbrumleve/house_price_prediction/blob/master/LICENSE
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: (https://www.linkedin.com/in/andrew-brumleve-574239227/)
[product-screenshot]: images/screenshot.png
[Python]:  	https://img.shields.io/badge/Python-14354C?style=for-the-badge&logo=python&logoColor=white
[Python-url]: https://python.org/
