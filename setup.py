# -*- coding: utf-8 -*-

from setuptools import setup, find_packages  
version = '0.1.3'

# pypandocを使ってREADME.mdをrstに変換する。最初からrstで書いた場合は不要。
try:  
    import pypandoc
    read_md = lambda f: pypandoc.convert(f, 'rst')
except ImportError:  
    print("warning: pypandoc module not found, could not convert Markdown to RST")
    read_md = lambda f: open(f, 'r').read()

setup(  
        name='stacking', # パッケージ名
        version=version, # バージョン
        description="A stacking library for ensemble learning", # 一言で説明
        # 詳細は http://pypi.python.org/pypi?:action=list_classifiers を参照
        classifiers=[
            "Development Status :: 3 - Alpha",
            "Environment :: Console",
            "Intended Audience :: Science/Research",
            "Natural Language :: English",
            "Operating System :: OS Independent",
            "Programming Language :: Python",
            "Topic :: Scientific/Engineering",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
            "Topic :: Utilities",
            "License :: OSI Approved :: MIT License",
            ],
        # キーワードを書いておく
        keywords='stacking, ensemble, machine learning, cross validation, sckit-learn, XGBoost, Keras, Vowpal Wabbit',
        # 作者の名前
        author="Ikki Tanaka",
        # 連絡先
        author_email="ikki0407@gmail.com",
        # GitHubのリポジトリとか
        url='https://github.com/ikki407/stacking',
        # ライセンス
        license='MIT',
        packages=find_packages(exclude=[]),
        include_package_data=True,
        zip_safe=True,
        # Pypiのページで表示する説明文(README)
        long_description=read_md('README.md'),
        # インストールする依存パッケージ
        install_requires=["numpy", "pandas", "sklearn", "xgboost", "keras"],
    )
