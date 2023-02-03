import setuptools

setuptools.setup(
    name='liboml',
    version='0.0.2',
    description='Library For Online Machine Learning',
    author="Welt Xing",
    author_email="xingcy@smail.nju.edu.cn",
    maintainer="Welt Xing",
    maintainer_email="xingcy@smail.nju.edu.cn",
    packages=['liboml', 'liboml/deep', 'liboml/kernel', 'liboml/linear'],
    license='MIT License',
    install_requires=['numpy', 'scikit-learn', 'torch'],
    long_description=open('README.md', encoding='utf-8').read(),
    long_description_content_type="text/markdown",
    url='https://github.com/Kaslanarian/LIBOML',
)