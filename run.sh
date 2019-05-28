echo "Downloading Datasets..."
wget https://home.strw.leidenuniv.nl/~nobels/coursedata/GRBs.txt
wget https://home.strw.leidenuniv.nl/~nobels/coursedata/randomnumbers.txt
wget https://home.strw.leidenuniv.nl/~nobels/coursedata/colliding.hdf5


echo "Run  scripts ..."
python3 q1.py
python3 q2.py
python3 q3.py
python3 q4.py
python3 q5.py
python3 q6.py
echo "Generating the pdf"

pdflatex handin_2.tex