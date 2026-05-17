const patterns = {
  rightTriangle: require('./patterns/rightTriangle'),
  invertedRightTriangle: require('./patterns/invertedRightTriangle'),
  pyramid: require('./patterns/pyramid'),
  invertedPyramid: require('./patterns/invertedPyramid'),
  diamond: require('./patterns/diamond'),
  hollowSquare: require('./patterns/hollowSquare'),
  hollowPyramid: require('./patterns/hollowPyramid'),
  numberPattern: require('./patterns/numberPattern')
};

function printSection(title, patternFn) {
  console.log(`\n${title}`);
  console.log('-'.repeat(title.length));
  console.log(patternFn(5));
}

function main() {
  console.log('js-star-patterns');
  console.log('A beginner-friendly collection of common star patterns.');

  printSection('Right Triangle', patterns.rightTriangle);
  printSection('Inverted Right Triangle', patterns.invertedRightTriangle);
  printSection('Pyramid', patterns.pyramid);
  printSection('Inverted Pyramid', patterns.invertedPyramid);
  printSection('Diamond', patterns.diamond);
  printSection('Hollow Square', patterns.hollowSquare);
  printSection('Hollow Pyramid', patterns.hollowPyramid);
  printSection('Number Pattern', patterns.numberPattern);
}

if (require.main === module) {
  main();
}

module.exports = { main, patterns };
