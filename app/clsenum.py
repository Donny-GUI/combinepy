import inspect


class ClassEnum:
    """
    A utility class to enumerate methods and attributes of a given class.
    """

    @classmethod
    def getMethods(cls, target_class) -> dict:
        """
        Retrieves all callable methods of a given class, categorized as:
        - Instance methods
        - Class methods
        - Static methods

        Args:
            target_class (type): The class to inspect.

        Returns:
            dict: A dictionary with keys 'instance_methods', 'class_methods', and 'static_methods',
                  each containing a list of corresponding method names.
        """
        # Retrieve instance methods
        instance_methods = [
            meth for meth, func in inspect.getmembers(target_class, predicate=inspect.isfunction)
        ]
        
        # Retrieve class and static methods
        all_methods = inspect.getmembers(target_class, predicate=inspect.ismethod)
        class_methods = [meth for meth, m in all_methods if isinstance(m.__func__, classmethod)]
        static_methods = [
            meth for meth, m in inspect.getmembers(target_class) if isinstance(m, staticmethod)
        ]

        return {
            'instance_methods': instance_methods,
            'class_methods': class_methods,
            'static_methods': static_methods,
        }

    @classmethod
    def getAttributes(cls, target_class) -> list:
        """
        Retrieves all attributes (both descriptors and non-descriptors) of a given class.

        Args:
            target_class (type): The class to inspect.

        Returns:
            list: A list of attribute names.
        """
        attributes = [
            attr for attr, _ in inspect.getmembers(target_class, predicate=inspect.isdatadescriptor)
        ]
        # Add non-descriptor attributes (e.g., instance variables or constants)
        non_descriptor_attributes = [
            attr for attr, value in inspect.getmembers(target_class)
            if not callable(value) and not attr.startswith("__")
        ]
        
        return attributes + non_descriptor_attributes
