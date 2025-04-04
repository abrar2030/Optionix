const path = require('path');  
const webpack = require('webpack');  

module.exports = {  
    entry: './src/index.js',  
    output: {  
        path: path.resolve(__dirname, 'dist'),  
        filename: 'bundle.js',  
        publicPath: '/'  
    },  
    module: {  
        rules: [  
            {  
                test: /\.(js|jsx)$/,  
                exclude: /node_modules/,  
                use: ['babel-loader']  
            },  
            {  
                test: /\.css$/,  
                use: ['style-loader', 'css-loader']  
            },  
            {  
                test: /\.(png|svg|jpg|gif)$/,  
                use: ['file-loader']  
            }  
        ]  
    },  
    plugins: [  
        new webpack.ProvidePlugin({  
            process: 'process/browser',  
        }),  
        new webpack.DefinePlugin({  
            'process.env.REACT_APP_API_BASE': JSON.stringify(process.env.REACT_APP_API_BASE)  
        })  
    ],  
    resolve: {  
        extensions: ['.js', '.jsx'],  
        fallback: {  
            "stream": require.resolve("stream-browserify"),  
            "crypto": require.resolve("crypto-browserify")  
        }  
    }  
};  