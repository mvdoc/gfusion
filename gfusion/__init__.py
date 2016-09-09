from .due import due, BibTeX
due.cite(
    BibTeX("""@inproceedings{zhang2014towards,
             title={Towards drug repositioning: a unified computational framework
             for integrating multiple aspects of drug similarity and disease similarity},
             author={Zhang, Ping and Wang, Fei and Hu, Jianying},
             booktitle={AMIA Annu Symp Proc},
             pages={1258--67},
             year={2014}
             }"""),
    description="Method to densify an association matrix",
    path='gfusion', version='0.0.1', cite_module=True)

