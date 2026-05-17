const { printRightTriangle } = require('./patterns/rightTriangle');
const { printPyramid } = require('./patterns/pyramid');
const { printDiamond } = require('./patterns/diamond');

function main() {
  console.log('js-star-patterns');
  console.log('=================\n');

  printRightTriangle(5);
  printPyramid(5);
  printDiamond(5);
}

if (require.main === module) {
  main();
}

module.exports = {
  main,
  printRightTriangle,
  printPyramid,
  printDiamond,
};
