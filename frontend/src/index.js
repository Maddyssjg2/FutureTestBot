const patterns = {
  rightTriangle: require('../patterns/rightTriangle'),
  invertedRightTriangle: require('../patterns/invertedRightTriangle'),
  pyramid: require('../patterns/pyramid'),
  invertedPyramid: require('../patterns/invertedPyramid'),
  diamond: require('../patterns/diamond'),
  hollowSquare: require('../patterns/hollowSquare'),
  hollowPyramid: require('../patterns/hollowPyramid'),
  numberPattern: require('../patterns/numberPattern'),
};

function runDemo() {
  console.log('js-star-patterns');
  console.log('=================\n');

  console.log('1) Right Triangle');
  console.log(patterns.rightTriangle(5));
  console.log('\n2) Inverted Right Triangle');
  console.log(patterns.invertedRightTriangle(5));
  console.log('\n3) Pyramid');
  console.log(patterns.pyramid(5));
  console.log('\n4) Inverted Pyramid');
  console.log(patterns.invertedPyramid(5));
  console.log('\n5) Diamond');
  console.log(patterns.diamond(5));
  console.log('\n6) Hollow Square');
  console.log(patterns.hollowSquare(5));
  console.log('\n7) Hollow Pyramid');
  console.log(patterns.hollowPyramid(5));
  console.log('\n8) Number Pattern');
  console.log(patterns.numberPattern(5));
}

if (require.main === module) {
  runDemo();
}

module.exports = { runDemo };
